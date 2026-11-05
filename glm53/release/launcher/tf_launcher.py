#!/usr/bin/env python3
"""tf_launcher.py [-c CONF] [--host H] [--port N] [--pkg DIR] [--dry-run]

The control panel's launcher: always on, on the lane's first node, beside tf-glm53.
  - GET /panel: the lane's page (TF_PANEL_HTML in the lane's EXTRA_ENV, else PKG/panel.html), read on each request
  - GET /v1/launch/status | nodes | usage, POST /v1/launch/plan | start | stop | resort
  - the saved allocations (ATLAS_DIR/allocs/user/<name>.json, glm53-expert-alloc v1): GET /v1/launch/allocs (the list,
    newest first, each with its counts, mean bits and whether its banks are this lane's) and /v1/launch/alloc?name=
    (one, with its bits), POST /v1/launch/allocs/save | rename | delete (moved to allocs/trash/); a start takes
    "alloc": a saved one's bits, banks and usage (status.alloc names it), or "usage_from": that one's usage with the bits
    sent (status.usage_from); a save with_usage keeps that usage + this run's picks, a re-sort with include_alloc ranks
    by them. Nothing is saved by itself unless asked: a re-sort with save false saves nothing
  - the banks: every width's manifest and layer files checked on every node (at start, then at most every 60 s, and
    again before a start): a width some node lacks is not offered (status.banks), makes a plan "no" (missing_bits)
    and refuses a start at once; a failed start shows the ranks' own error lines
  - the drafter: a plan or a start takes `drafter` (mtp, dflash2, none; CONF's DRAFTER_KIND when absent) and runs
    tf-glm53 -d with it; DFlash2 only when DRAFTER holds its model on every node (checked with the banks;
    status.drafter); the memory of a plan from the last measured start with that drafter (launch/last-start-<d>.json)
  - a start's context: plan and start take `context` (positions, a multiple of 16,384, at least 65,536; CONF's when
    absent), cut when it does not fit; kv.formats.<f>.max_context the longest that fits (at most 1,048,576)
  - a start's store and reserve: plan and start take `store` (tokens, or "max": the rest of the KV (a multiple of 16,384)
    room)) and `reserve_gib` (what a node keeps for the OS, 2 to 32; RESERVE_GB of CONF, 4 when absent); the engine's
    admission gets the same reserve and tf-glm53 check the plan's need (LAUNCH_NEED_GB)
  - the nodes (the panel's node check, its first screen until the nodes are registered): GET /v1/launch/config (NODES
    in rank order, the topology: full, or ring4 with four nodes only; registered: whether the node file exists);
    POST /v1/launch/discover (GET: its result) reads how the head is cabled: a ring (LLDP), or a switch whose hosts on
    the head's CX7 /24 accept ssh's port (only those /24s are swept; the management LAN never is), each read over
    ssh; POST /v1/launch/check {"head", "workers"} runs tf-glm53 deploy and check on them with TOPOLOGY=auto (each
    node's items, the ring's check on a ring, the cabling judged: topology); POST /v1/launch/config {"head",
    "workers"} writes the node file (glm53.nodes.conf beside CONF: TOPOLOGY, NODES, SOCKET_IFNAME, IB_HCA as judged;
    read after CONF by tf-glm53 and here, so it wins; the previous one copied aside; only while stopped) once that
    list passed
  - where the weights are (the launch settings): GET /v1/launch/weights (WEIGHTS, DRAFTER, every bank's directory),
    POST /v1/launch/weights/check (the locations on every node), POST /v1/launch/weights (once they passed: glm53.conf's
    WEIGHTS and DRAFTER lines and banks.json's directories written, each copied aside first; banks.json copied to every
    worker; only while stopped)
  - the engine settings (R-024, the panel's engine settings screen): GET /v1/launch/settings (glm53.conf's keys, SETTINGS: each
    one's value as tf-glm53 reads it, its default, rule and when a change is used; the keys changed since the running
    start), POST /v1/launch/settings (the changed keys checked as tf-glm53 checks them, then their lines written, the
    conf copied aside first), POST /v1/launch/restart (a stop, then a start with the bits it ran and the saved
    settings); the launcher reads a changed conf again by itself while the engine is stopped
  - GET / POST /v1/panel/settings: the panel's settings, one JSON document in a file of the lane (TF_PANEL_SETTINGS
    in EXTRA_ENV, else PKG/launch/panel-settings.json), served here whether the engine runs or not; not read, replaced whole
  - every other path goes to the engine (HOST:PORT of glm53.conf), 503 while it is not running: the page has one
    address whether the engine runs or not
  - access: anyone who reaches LAUNCHER_HOST:LAUNCHER_PORT (no key; every address by default: keep the nodes on a closed
    network, or LAUNCHER_HOST=127.0.0.1 and an ssh tunnel); a start or a stop at a time (409)
A start: the pre-flight (every node reachable, its GPU readable by nvidia-smi, no engine or other GPU process on any node, the plan fits; with
LAUNCH_SNAPSHOTS=1 the nodes' state written down, with LAUNCH_DROP_CACHES=1 the page cache dropped), the bits written
as an expert plan (tools/quant/expert_plan.py mix with expert-plan/banks.json), then `tf-glm53 engine-up` with LAUNCH_ENV="TF_GLM53_EXPERT_PLAN=... TF_GLM53_LOAD_PROGRESS=...";
the engine's progress file (glm53-load-progress v1) feeds /v1/launch/status. The ranges (the counts by bits) stay as
the start set them; POST /v1/launch/resort re-sorts the experts between them while the engine runs (the generic kernel,
the swap banks) and saves the allocation, by the decode picks, the prompt picks or both (basis; both weighs decode 0.7, prompt 0.3). A stop: the
usage kept, `tf-glm53 engine-down`, then the run's decode and prompt picks saved into the learning data, dropped, or kept
pending (save_usage true / false / absent; /v1/launch/usage/save
and /discard decide the pending). Kept in PKG/launch/: the learning data (usage-total.json), the pending runs, the last
run, the last measured start (the plan's sizes and the ETA), the plans; allocations in ATLAS_DIR/allocs/user; the
snapshots (LAUNCH_SNAPSHOTS=1) in PKG/snapshots/.
Defaults: CONF = glm53.conf beside this script's directory (PKG = the lane directory), port LAUNCHER_PORT of CONF
(else PORT + 10), host LAUNCHER_HOST (else 0.0.0.0: every address). --pkg: the lane directory on the nodes when this runs elsewhere
(a test from another machine). --dry-run: a start or a stop runs its read-only checks, prints what it would run, and runs
nothing that changes a node.
"""
import argparse, base64, hashlib, ipaddress, json, os, re, select, shlex, shutil, signal, socket, struct, subprocess, sys, threading, time
import http.client
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

GiB = 1 << 30
# GLM-5.3: hidden D, routed expert intermediate I, experts a layer, the first MoE layer, MoE layers
D, I, E, FIRST, L = 6144, 2048, 256, 3, 75
# the stored shares of each routed expert (the checkpoints keep them cut for TP=8: TF_GLM53_STORED_WORLD)
S = 8
# its layer (picks / the layer's picks), averaged; a source with no rows drops out
BASES = ("decode", "prompt", "both")
# "both": an expert's decode share weighed 0.7 and its prompt share 0.3 (the long-prompt KLD test, ATLAS-TESTS.md:
# the text written gains with the decode weight up to about 0.7, the documents read hardly depend on it)
DECODE_WEIGHT = 0.7
# a saved allocation's name (ATLAS_DIR/allocs/user/<name>.json) and its note's longest
ALLOC_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
NOTE_MAX = 500
# the panel's settings: the largest document a POST may carry
SETTINGS_MAX = 64 << 10
BANK_TTL = 60
# run on each node: every width's bank (its manifest and the layer files it lists, present at the manifest's size)
BANK_PROBE = r'''
import json, os, sys, base64
out = {}
a = json.loads(base64.b64decode(sys.argv[1]))
if a.get("drafter"):  # the DFlash2 model of DRAFTER, on this node
    d = os.path.expanduser(a["drafter"])
    out["dflash2"] = "ok" if os.path.isfile(os.path.join(d, "config.json")) else "no model at " + a["drafter"]
for b in a["banks"]:
    try:
        m = json.load(open(b["manifest"]))
        layers = m["layers"]
    except (OSError, ValueError, KeyError) as e:
        out[str(b["bits"])] = "no manifest " + b["manifest"]
        continue
    miss = bad = 0
    for x in layers:
        try:
            if os.stat(os.path.join(b["dir"], x["file"])).st_size != x["size"]:
                bad += 1
        except OSError:
            miss += 1
    out[str(b["bits"])] = "ok" if miss == bad == 0 else "%d of %d layer files missing, %d of another size in %s" % (miss, len(layers), bad, b["dir"])
print(json.dumps(out))
'''
# run on each node: where the weights are (POST /v1/launch/weights/check): the checkpoint (config.json and
# model.safetensors.index.json), DFlash2's model (config.json), each bank row (a width: its manifest found by four
# rules, its bits the row's, the layer files it lists at the manifest's size; nothing found and no directory: missing,
# warn). One JSON line, {key: [state, value]} ({key: [state, value, the manifest's facts]} for a bank row)
WEIGHTS_PROBE = r"""
import base64, glob, hashlib, json, os, socket, sys
a = json.loads(base64.b64decode(sys.argv[1]))
x = os.path.expanduser
out = {"_host": socket.gethostname()}
w = x(a["weights"])
if not os.path.isdir(w):
    out["weights"] = ["fail", "no directory " + a["weights"]]
else:
    miss = [f for f in ("config.json", "model.safetensors.index.json") if not os.path.isfile(os.path.join(w, f))]
    n = len([f for f in os.listdir(w) if f.endswith(".safetensors")])
    out["weights"] = ["fail", "no " + ", ".join(miss) + " in " + a["weights"]] if miss else ["ok", "%d safetensors" % n]
if a["drafter"]:
    d = x(a["drafter"])
    out["drafter"] = ["ok", "config.json"] if os.path.isfile(os.path.join(d, "config.json")) else ["fail", "no model at " + a["drafter"]]
# a row's bank: its manifest (banks.json's while the directory is its own; <dir>/<id>.json; the one bank manifest in
# the directory; a manifest of <dir>/../expert-banks/ made from the directory), parsed once a file
seen = {}
def load(p):
    q = x(p)
    if q not in seen:
        seen[q] = (None, None)
        try:
            raw = open(q, "rb").read()
            m = json.loads(raw)
            if isinstance(m, dict) and isinstance(m.get("layers"), list):
                seen[q] = (m, hashlib.sha256(raw).hexdigest())
        except (OSError, ValueError):
            pass
    return seen[q]
NOT_BANK = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "model.safetensors.index.json", "chat_template.json", "preprocessor_config.json"}
def find(b):
    d = b["dir"]
    if b["bj"]:
        m, h = load(b["bj"])
        if m:
            return m, h, b["bj"], "banks.json"
    p = os.path.join(d, b["row"] + ".json")
    m, h = load(p)
    if m:
        return m, h, p, "dir/id.json"
    for where, how, ok in ((d, "in dir", lambda m: True),
                           (os.path.join(os.path.dirname(d.rstrip("/")), "expert-banks"), "expert-banks",
                            lambda m: isinstance(m.get("made_from"), str) and os.path.realpath(x(m["made_from"])) == os.path.realpath(x(d)))):
        hits = []
        for f in sorted(glob.glob(os.path.join(x(where), "*.json"))):
            if os.path.basename(f) in NOT_BANK:
                continue
            m, h = load(f)
            if m and m.get("format") == "glm53-expert-bank" and ok(m):
                hits.append((m, h, os.path.join(where, os.path.basename(f)), how))
        if len(hits) > 1:
            return None, None, "%d bank manifests %s" % (len(hits), "in " + d if how == "in dir" else "in %s made from %s" % (where, d)), "several"
        if hits:
            return hits[0]
    return None, None, None, None
for b in a["banks"]:
    k, d = "bank:" + b["row"], x(b["dir"])
    if not b["dir"]:  # no directory: that width left unused (11-01)
        out[k] = ["warn", "not used", None]
        continue
    m, h, man, how = find(b)
    if how == "several":
        out[k] = ["fail", man, None]
        continue
    if m is None:
        out[k] = ["fail", "no bank manifest for " + b["dir"], None] if os.path.isdir(d) else ["warn", "not here", None]
        continue
    L = m["layers"]
    sizes = [f.get("size") for f in L]
    info = {"id": m.get("id"), "bits": m.get("bits"), "manifest": man, "manifest_sha256": h, "how": how,
            "bytes": sum(sizes) if all(isinstance(v, int) for v in sizes) else None, "calibration": m.get("calibration"),
            "codebook": m.get("codebook"), "layer_count": m.get("layer_count", len(L)), "first_layer": m.get("first_layer"),
            "stored_ranks": m.get("stored_ranks")}
    if m.get("bits") != b["bits"]:
        out[k] = ["fail", "this bank is %s bit" % m.get("bits"), info]
        continue
    miss = bad = 0
    for f in L:
        try:
            if os.stat(os.path.join(d, f["file"])).st_size != f["size"]:
                bad += 1
        except (OSError, KeyError, TypeError):
            miss += 1
    out[k] = (["ok", "%s · %d layer files" % (m.get("id"), len(L)), info] if miss == bad == 0
              else ["fail", "%d of %d layer files missing, %d of another size" % (miss, len(L), bad), info])
print(json.dumps(out))
"""
# a location in glm53.conf or banks.json as the launch settings set it: ~/... or /..., written unquoted
WPATH = re.compile(r"(?:~/|/)[A-Za-z0-9._@+/-]*")
# the launch settings' directory dialog (GET /v1/launch/browse): the head's directories under these places only (the
# panel listens on the network without a key), hidden ones left out, at most BROWSE_MAX a listing
BROWSE_ROOTS = ("~", "/opt", "/mnt", "/media", "/data", "/srv")
BROWSE_MAX = 500
MEMORY_ONLY = {"last-usage.json", "pending.json", "usage-base.json"}  # kept in the launcher's memory, never written
TESTED_WORLDS = (2, 4, 8)  # the node counts the panel has run on, offered by GET /v1/launch/config (worlds)
# the check's items that may fail on nodes a node check registers all the same (nodes_ok): where the weights are,
# set afterwards on the launch settings (the checkpoint, the expert banks)
NODES_OK_FAILS = ("weights", "banks")
# the engine's rule: the 3-bit bank must be on every node, whatever a start uses, and a start that names no
# widths loads every expert at 3 bits
REQUIRED_BITS = 3
CHECKPOINT_REPO = {4: "fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks", 8: "fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks"}  # on Hugging Face
# the order of a node's check items (tf-glm53 check with CHECK_ITEMS, banks from the launcher)
def lan_name():
    """This node's address on the default route (what another machine on the LAN opens), else its name."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


CHECK_ORDER = ("ssh", "gpu", "memory", "rendezvous", "rdma", "image", "nccl", "weights", "banks", "versions", "package")
# the node counts this engine runs: each expert's 8 stored shares joined 8 / world a rank (2, 4, 8 nodes). Three nodes
# (TP=3: the shares themselves as 3 / 3 / 2 units, heads 22 / 21 / 21) are planned here but need an engine that splits
# the shares so: a start on them is refused until it has one
ENGINE_WORLDS = (2, 4, 8)
# the longest context a start may ask (the engine and the model take 1,048,576 positions); a context is a multiple of
# CONTEXT_STEP positions, at least CONTEXT_MIN
CONTEXT_LIMIT, CONTEXT_STEP, CONTEXT_MIN = 1048576, 16384, 65536
ADDR = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:_-]{0,252}")  # a node's address in NODES (a host name or an IP)
EXAMPLE_ADDR = "192.0.2."  # the example conf's NODES before the first node check: a range reserved for examples (RFC 5737)
RESERVE = 4 * GiB  # what a node keeps for the OS after the start, by default (RESERVE_GB of CONF, or a plan's reserve_gib);
# the engine's own admission gets the same (TF_GLM53_MEM_RESERVE_GIB)
RESERVE_MIN_GIB, RESERVE_MAX_GIB = 2, 32
ESTIMATE_SLACK = 2 * GiB  # kept beside the reserve when the store takes the rest on an estimate (no measured start)
# kept beside the reserve when the store takes the rest on a measured start: the memory moves a little between the plan
# and the start's check (a plan that took it all failed the check by 0.1 to 0.3 GiB)
MEASURED_SLACK = 512 << 20
WORK_SLOPE_MAX = 64 << 10  # bytes of work a position at most, from two measured starts (a guard against a bad pair)
START_HISTORY = 8  # measured starts kept a drafter (their positions and work), for the work's growth with the context
POOL_MIN_STORE = 512  # the least store with CACHE=pool: the engine needs the context + 8 rows + a block (at most 264)
# the drafters a start may take (tf-glm53 -d): the checkpoint's MTP layer, DFlash2 (DRAFTER of CONF, on every node), none
DRAFTERS = ("mtp", "dflash2", "none")
# the conversations decoded together (MULTI of CONF, 0 .. MULTI_MAX; a start may take another: multi, as tf-glm53's
# LAUNCH_MULTI), DESIGN-multi-start.md: a start with MULTI >= 1 holds MULTI_IDLE more a node (window buffers, MoE
# scratch, the MTP stage, the per-conversation tables, graphs: N=4 measured 0.55-1.0 GiB on TP4, N 2-3 not measured),
# and N >= 2 conversations at once take MULTI_LOAD[world][drafter] GiB more under load (the top of CONCURRENT.md
# §15.6 / §15.7's ranges); both before the KV, so the reserve holds under load (10-30). No drafter takes MTP's,
# other node counts four nodes' and N 2-3 N=4's: estimates. ring4: two conversations at most until a ring test
# passed (10-30); all four offered since (after two at once ran on the ring)
MULTI_MAX = 4
MULTI_WAIT_DEFAULT = 600
MULTI_IDLE = 1 * GiB
MULTI_LOAD = {4: {"mtp": 1.5, "dflash2": 2.2, "none": 1.5}, 8: {"mtp": 1.3, "dflash2": 1.9, "none": 1.3}}
# measured by the conversations, in place of the two above: {(world, drafter): {N: (idle, load)}} GiB over MULTI=0, the
# largest rank, rounded up (four nodes on a switch, 262,144 + 262,144 positions, N conversations of 100K tokens). MTP's
# load is what thirty rounds at N=4 took (rank 0 grew 1.32 GiB and stopped); N 2 and 3 not run so long
MULTI_BY_N = {(4, "dflash2"): {1: (0.6, 0), 2: (0.6, 0.6), 3: (0.7, 0.7), 4: (0.9, 0.9)},
              (4, "mtp"): {1: (0.2, 0), 2: (0.2, 1.2), 3: (0.2, 1.3), 4: (0.3, 1.3)}}
MULTI_MEASURED = {(4, "mtp"), (4, "dflash2"), (8, "mtp"), (8, "dflash2")}
MULTI_RING_MAX = 4
# the engine settings screen (R-024): the glm53.conf keys the
# panel shows and writes, each as (tf-glm53's default, the rule, when a change is used). The rules are tf-glm53's checks
# (the launcher refuses the same values first); the panel's own checks read them. when: start (saved any time, used
# from the next start), live (saved any time, and a running engine takes it at once: SET_LIVE), stopped (saved only
# while the engine is stopped: tf-glm53 down finds the ranks by PORT),
# launcher (the launcher's own, used once the launcher is started again), readonly (shown, changed elsewhere: WEIGHTS
# and DRAFTER on the [ WEIGHTS ] card, the nodes by the node check). TF_GLM53_CANON_* are tokens of EXTRA_ENV (the
# engine reads them from its environment). DCP's options and LAUNCHER_PORT's default follow the lane (es_table)
SET_INT = lambda lo, hi, **k: dict({"type": "int", "min": lo, "max": hi}, **k)
SET_TEXT = lambda pat, most=255: {"type": "text", "pattern": pat, "max": most}
SET_ENUM = lambda *o: {"type": "enum", "options": list(o)}
SET_BOOL = {"type": "bool"}
SET_BIG = 1 << 24  # the most tokens a count may name (tf-glm53 has no limit above: a guard against typos)
SET_IPV4 = r"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])$"
SET_PROMPT_DEFAULT = "Organize your thoughts so far. Keep code, references and numbers exactly as they are."
SET_PROMPT_RE = r"[^\x00-\x08\x0b-\x1f\x7f]*"
SET_ENV = r"^(?:[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9._:/,=+@%~-]*(?: +(?=[A-Za-z_])|$))*$"  # EXTRA_ENV: K=V tokens
SETTINGS = {
    "NAME": ("glm-5.3", SET_TEXT(r"^[A-Za-z0-9._:/@+-]{1,128}$", 128), "start"),
    "WEIGHTS": ("", {"type": "text"}, "readonly"),
    # one request's longest and the whole KV (11-02, vLLM's wording); CONTEXT and STORE, the names before, read as them
    # (KV_SIZE then CONTEXT + STORE) and are dropped when these are written
    "CONTEXT_LENGTH": ("819200", SET_INT(CONTEXT_MIN, CONTEXT_LIMIT, step=CONTEXT_STEP), "start"),
    "KV_SIZE": ("1638400", SET_INT(CONTEXT_MIN, SET_BIG), "start"),
    "KV": ("fp8", SET_ENUM("fp8", "fp4"), "start"),
    "CACHE": ("pool", SET_ENUM("pool"), "start"),
    "MAX_TOKENS": ("", SET_INT(1, SET_BIG, empty=True), "start"),
    "MAX_TOKENS_SHARE": ("70", SET_INT(1, 100), "start"),
    "THINKING_BUDGET": ("0", SET_INT(0, SET_BIG), "start"),
    "THINKING_SHARE": ("90", SET_INT(0, 100), "live"),
    "THINKING_FOLD": ("off", SET_ENUM("off", "checkpoint", "recent", "branch"), "live"),
    # the fold comes when the thinking fills its share of the context (11-02); these, above 0, make it come earlier
    "THINKING_FOLD_SHARE": ("100", SET_INT(10, 100), "live"),
    "THINKING_FOLD_MAX": ("0", SET_INT(0, 1000), "live"),
    "THINKING_FOLD_AT_BUDGET": ("1", SET_INT(0, 1), "live"),
    "THINKING_FOLD_MARGIN": ("10", SET_INT(0, 50), "live"),
    # the texts the model is asked, plain here and in the API, percent-encoded in the conf (a shell script); empty: back
    # to the default, the same text as the engine's server.zig
    "THINKING_FOLD_PROMPT": (SET_PROMPT_DEFAULT, SET_TEXT(SET_PROMPT_RE, 4096), "live"),
    "THINKING_CLOSE_PROMPT": (SET_PROMPT_DEFAULT, SET_TEXT(SET_PROMPT_RE, 4096), "live"),
    "THINKING_FOLD_CAP_SHARE": ("70", SET_INT(10, 100), "start"),
    "THINKING_FOLD_AT_CLOSE_SHARE": ("0", SET_INT(0, 100), "live"),
    "THINKING_NOTICE": ("0", SET_INT(256, SET_BIG, zero=True), "start"),
    "MULTI": ("0", SET_INT(0, MULTI_MAX), "start"),
    "MULTI_WAIT": ("600", SET_INT(0, 86400), "start"),
    "MULTI_DECODE_SHARE": ("30", SET_INT(0, 100), "start"),
    "MULTI_SHORT_PROMPT": ("8192", SET_INT(0, 1048576), "start"),
    "MULTI_SHORT_PREEMPT": ("0", SET_BOOL, "start"),
    "DRAFTER_KIND": ("mtp", SET_ENUM(*DRAFTERS), "start"),
    "MTP_K": ("3", SET_INT(1, 7), "start"),
    "DRAFT_SAVES": ("8", SET_INT(0, 64), "start"),
    "DRAFTER": ("", {"type": "text"}, "readonly"),
    "NODES": ("", {"type": "text"}, "readonly"),
    "TOPOLOGY": ("full", SET_ENUM("full", "ring4", "auto"), "start"),
    "SOCKET_IFNAME": ("", SET_TEXT(r"^[A-Za-z0-9._@-]{1,15}$", 15), "start"),
    "IB_HCA": ("", SET_TEXT(r"^[A-Za-z0-9_.-]{1,32}(?:,[A-Za-z0-9_.-]{1,32})*$"), "start"),
    "DCP": ("2", SET_ENUM("2"), "start"),
    "MASTER_PORT": ("29551", SET_INT(1, 65535), "start"),
    "RESERVE_GB": ("4", SET_INT(RESERVE_MIN_GIB, RESERVE_MAX_GIB), "start"),
    "READY_TIMEOUT": ("900", SET_INT(60, 7200), "start"),
    "LAUNCH_DROP_CACHES": ("0", SET_BOOL, "start"),
    "HOST": ("0.0.0.0", SET_TEXT(SET_IPV4, 15), "start"),
    "PORT": ("8000", SET_INT(1, 65535), "stopped"),
    "METRICS_LISTEN": ("", SET_TEXT(r"^(?:(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+):[0-9]{1,5})?$", 64), "start"),
    "LOG_KEEP_MB": ("8", SET_INT(1, 1024), "start"),
    "LAUNCHER_PORT": ("8010", SET_INT(1, 65535), "launcher"),
    "ROWS": ("8192", SET_INT(256, 65536, step=256), "start"),
    "SP_PARTS": ("4", SET_INT(2, 8), "start"),
    "ROCE_PROXY_CORES": ("a725", SET_ENUM("a725", "os"), "start"),
    "ATLAS_GENERIC": ("1", SET_BOOL, "start"),
    "TF_GLM53_CANON_CLOSE": ("near", SET_ENUM("near", "all", "off"), "start"),
    "TF_GLM53_CANON_NEAR": ("4096", SET_INT(1, 1048576), "start"),
    "TF_GLM53_BYTE_RESUME": ("on", SET_ENUM("off", "log", "on"), "start"),
    "EXTRA_ENV": ("", SET_TEXT(SET_ENV, 4096), "start"),
}
SET_CANON = ("TF_GLM53_CANON_CLOSE", "TF_GLM53_CANON_NEAR", "TF_GLM53_BYTE_RESUME")  # EXTRA_ENV's tokens with keys of their own
# when "live": the thinking defaults the running engine takes at once (POST /v1/thinking, R-029), each key's field
# there. A save of them while the engine runs is sent to it too, and the inference screen's [ THINKING ] (POST
# /v1/thinking through the launcher) is written to the conf too, so both screens show one value
SET_LIVE = {"THINKING_SHARE": "share", "THINKING_FOLD": "fold", "THINKING_FOLD_AT_CLOSE_SHARE": "at_close_share", "THINKING_FOLD_SHARE": "fold_share", "THINKING_FOLD_MAX": "fold_max", "THINKING_FOLD_AT_BUDGET": "at_budget", "THINKING_FOLD_MARGIN": "fold_margin", "THINKING_FOLD_PROMPT": "fold_prompt", "THINKING_CLOSE_PROMPT": "close_prompt"}
SET_PROMPTS = ("THINKING_FOLD_PROMPT", "THINKING_CLOSE_PROMPT")
SET_LIVE_TEXT = ("fold", "fold_prompt", "close_prompt")
# read besides SETTINGS' keys: THINKING_FOLD_KEEP, a conf's own;
# CONTEXT and STORE for CONTEXT_LENGTH and KV_SIZE when a conf names only the old keys (SET_OLD)
SET_HIDDEN = ("THINKING_FOLD_KEEP", "CONTEXT", "STORE")
SET_OLD = ("CONTEXT", "STORE")
# the keys the node check writes once it registered the nodes (the node file wins): read only here then
SET_NODE_FILE = ("TOPOLOGY", "SOCKET_IFNAME", "IB_HCA")
# a start's own choices and the conf keys they default to (POST /v1/launch/restart drops a choice whose key changed in
# the conf since the start: the conf's value then counts)
SET_ASKED = {"kv": "KV", "drafter": "DRAFTER_KIND", "context": "CONTEXT_LENGTH", "store": "KV_SIZE", "reserve_gib": "RESERVE_GB",
             "multi": "MULTI", "multi_wait": "MULTI_WAIT"}
# the variables a ring needs that EXTRA_ENV may not set (tf-glm53's list, TOPOLOGY=ring4)
RING_ENV = ("NCCL_ALGO NCCL_PROTO NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_ADDR_FAMILY NCCL_IB_ROCE_VERSION_NUM NCCL_IB_MERGE_NICS "
            "NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_SUBNET_PREFIX_LEN NCCL_CROSS_NIC NCCL_RUNTIME_CONNECT NCCL_CUMEM_ENABLE "
            "NCCL_PAT_ENABLE NCCL_NVLS_ENABLE NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_SOCKET_IFNAME TF_NCCL_LIB "
            "TF_ROCE_PROXY_LIB B12X_ROCE_HCA B12X_ROCE_GID_INDEX TF_GLM53_TOPOLOGY TF_GLM53_PREFILL_REDUCE TF_GLM53_MULTI_AR2 "
            "TF_GLM53_DCP TF_GLM53_WORLD TF_GLM53_RANK TF_GLM53_MASTER").split()
# the sizes a rank holds besides its routed experts, by node count and drafter, the largest node measured: used until
# this lane has a measured start with that drafter and node count (estimate: true). MTP holds its layer in the other
# weights; DFlash2 its model and state in the work, on every rank alike. Node counts not listed take four nodes'.
#   4 nodes: 737,280 + 100,000 positions, fp4, DCP 2
#   8 nodes (the release test): MTP 819,200 + 819,200 positions, DFlash2 1,048,576 + 704,512, fp8, DCP 2; no drafter
#     as DFlash2's other weights and MTP's work
#   2 nodes: 65,536 + 512 positions, fp8, DCP 2, ROWS 4,096 in 2 SP parts, 2-bit experts; DFlash2 not run there: its
#     work is three nodes' DFlash2 share x 1.5, an estimate
#   3 nodes: 131,072 + 32,768 positions, fp8, DCP 3, CACHE=slots, PEER_Q, 3-bit experts, measured with an engine that
#     splits the shares in three
G = lambda x: int(x * GiB)
DEFAULTS = {4: {"mtp": {"other_weights": G(8.11), "kv": G(14.55), "work": G(12.8)},
                "dflash2": {"other_weights": G(6.515), "kv": G(14.323), "work": G(17.42)},
                "none": {"other_weights": G(6.515), "kv": G(14.323), "work": G(12.8)}},
            8: {"mtp": {"other_weights": G(5.625), "kv": G(41.962), "work": G(8.413)},
                "dflash2": {"other_weights": G(4.704), "kv": G(44.246), "work": G(13.619)},
                "none": {"other_weights": G(4.704), "kv": G(44.246), "work": G(8.413)}},
            2: {"mtp": {"other_weights": G(11.52), "kv": G(1.70), "work": G(6.74)},
                "dflash2": {"other_weights": G(8.92), "kv": G(1.67), "work": G(11.3)},
                "none": {"other_weights": G(8.92), "kv": G(1.67), "work": G(6.47)}},
            3: {"mtp": {"other_weights": G(10.57), "kv": G(2.24), "work": G(6.9)},
                "dflash2": {"other_weights": G(8.69), "kv": G(2.24), "work": G(10.1)},
                "none": {"other_weights": G(8.69), "kv": G(2.24), "work": G(6.9)}}}
PHASES = ["checking", "weights", "kv", "prewarm", "ready"]
# the KV cache's bytes a row on one rank (forward.zig State.bytes): every cache layer's latent row (fp4 432 B, fp8 656 B)
# and the full-indexer layers' index row (144 B); GLM-5.3 with DFlash2: 78 cache layers, 21 of them with an indexer.
# A position takes a row on one rank of its DCP group: bytes a position = row / DCP.
KV_LAYERS, KV_INDEX_LAYERS, KV_LATENT, KV_INDEX = 78, 21, {"fp4": 432, "fp8": 656}, 144


# DFlash2's model is BF16 on disk; the engine quantizes its linear layers to 4 bits on the host at the start
# (zig/src/families/glm_moe_dsa/draft_weights.zig, q4.zig), its norms and conv base kernels staying BF16
DFLASH2_BITS = 4


def weights_info(wdir, drafter):
    """R-018: the bits beside the routed experts as the checkpoint says them: its config.json's quantization_config
    (other_linears' bits; unquantized, the tensors left BF16) and its MTP files' names (mtp-layer-N-experts-Kbit,
    -linears-Kbit); DFlash2's as the engine loads it (DFLASH2_BITS) when DRAFTER holds its model. None when the
    checkpoint does not say."""
    try:
        q = json.load(open(os.path.join(wdir, "config.json"))).get("quantization_config") or {}
        names = os.listdir(wdir)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    bits = (q.get("other_linears") or {}).get("bits") if isinstance(q, dict) else None
    if not isinstance(bits, int) or isinstance(bits, bool):
        return None
    un = q.get("unquantized")
    bf16 = [x.split(":")[0].strip() for x in un.split(",")] if isinstance(un, str) else [str(x) for x in un] if isinstance(un, list) else []
    out = {"nonexpert_bits": bits, "bf16": [x for x in bf16 if x], "drafters": {}}
    def named(kind):
        for n in names:
            m = re.match(rf"mtp-layer-\d+-{kind}-(\d)bit\.safetensors$", n)
            if m:
                return int(m.group(1))
        return None
    ex, li = named("experts"), named("linears")
    if ex is not None and li is not None:
        out["drafters"]["mtp"] = {"expert_bits": ex, "other_bits": li}
    if drafter and os.path.isfile(os.path.join(os.path.expanduser(drafter), "config.json")):
        out["drafters"]["dflash2"] = {"bits": DFLASH2_BITS}
    return out


def start_entry(doc):
    """A measured start as the plans' history keeps it: its positions (context + store) and each node's work."""
    kv = doc.get("kv") or {}
    return {"positions": int(kv.get("context") or 0) + int(kv.get("store") or 0), "unix": doc.get("unix", 0),
            "work": {n: v.get("work", 0) for n, v in (doc.get("nodes") or {}).items()}}


def manifest_of(bank):
    """A bank's manifest: banks.json's `manifest`, else <id>.json in the bank's directory (as the engine looks)."""
    return bank.get("manifest") or os.path.join(bank["dir"], bank["id"] + ".json")


def log(msg):
    print(time.strftime("%F %T ") + msg, file=sys.stderr, flush=True)


def defaults(world, drafter):
    """DEFAULTS for `world` nodes and `drafter` (four nodes' for a node count not measured)."""
    return DEFAULTS.get(world, DEFAULTS[4])[drafter]


def experts_bytes(bits, world):
    """The routed experts' bytes on one rank, as the engine counts them (planBytes). A world dividing the 8 stored
    shares joins 8 / world of them an expert: one matrix I / world wide. Three nodes hold the shares themselves, 2 or 3
    of an expert (unit_shares), each a matrix I / 8 wide, with the expert's scales once and every expert's 3 unit
    slots' scales and pointers whether held or not (the three-way engine's unit load): the most any rank holds."""
    if S % world == 0:
        ir = I // world
        per = {b: 3 * ((D * ir * b // 8 + 255) & ~255) + 6 * (D + ir) + 36 for b in set(bits)}
        return sum(per[b] for b in bits)
    u, upe = I // S, -(-S // world)
    unit = {b: 3 * ((D * u * b // 8 + 255) & ~255) for b in set(bits)}
    held = [sum(unit_shares(i % E, FIRST + i // E, r, world) * unit[b] for i, b in enumerate(bits)) for r in range(world)]
    return max(held) + len(bits) * (6 * D + upe * (6 * u + 40))


def unit_shares(e, layer, rank, world):
    """The stored shares of expert `e` of `layer` that rank `rank` holds over `world` (split.zig unitShares on
    the three-way engine): S // world each, the S % world extra ones on the ranks from (e + layer + 1) % world on."""
    base, rem = S // world, S % world
    return base + ((rank + world - (e + layer + 1) % world) % world < rem)


def split_problem(world, dcp, cache):
    """Why CONF's node count, DCP and CACHE cannot run together (tf-glm53 and the engine refuse the same), or ""."""
    if world % dcp:
        return f"DCP={dcp} does not divide {world} nodes"
    if cache == "pool" and dcp > 2:
        return f"CACHE=pool takes DCP 1 or 2, not {dcp} (CACHE=slots, or DCP=1)"
    return ""


def nodes_file_of(conf):
    """The node file beside CONF, as tf-glm53 reads it: NODES_CONF when set (empty: none), else ${CONF%.conf}.nodes.conf
    (glm53.conf: glm53.nodes.conf). The control panel's node check writes it (save_config); its lines win over CONF's."""
    if "NODES_CONF" in os.environ:
        return os.environ["NODES_CONF"]
    return (conf[:-5] if conf.endswith(".conf") else conf) + ".nodes.conf"


# what glm53.conf, a shell script every command sources, holds as plain text: nothing a shell would run or expand
SET_LINE_VALUE = re.compile(r'[A-Za-z0-9_.:/,=+@%~\[\]-]*|"[A-Za-z0-9_.:/,=+@%~ -]*"')


def set_line(text, key, value):
    """glm53.conf's text with KEY=value: its one KEY= line replaced (a comment after the value kept), or appended when
    it has none."""
    if not SET_LINE_VALUE.fullmatch(str(value)):
        raise Fail(f"{key}: not a value glm53.conf can hold as plain text")
    pat = re.compile(rf"^{key}=(?:[^#\n]*?)(\s+#.*)?$", re.M)
    n = len(pat.findall(text))
    if n > 1:
        raise Fail(f"{key} must be one line of glm53.conf, {key}=...")
    if n == 1:
        return pat.sub(lambda m: f"{key}={value}" + (m.group(1) or ""), text)
    return text + ("" if text.endswith("\n") or not text else "\n") + f"{key}={value}\n"


def drop_lines(text, keys):
    """glm53.conf's text without its KEY= lines for `keys` (commented ones kept)."""
    return re.sub(r"^(?:" + "|".join(map(re.escape, keys)) + r")=.*\n?", "", text, flags=re.M)


def write_whole(path, text, keep=True):
    """A settings file written whole (a temporary file, then a rename), the previous one copied aside first as
    PATH.bak-YYYYMMDD-HHMMSS; the copy's name ("" when there was none)."""
    bak = ""
    if keep and os.path.exists(path):
        bak = path + time.strftime(".bak-%Y%m%d-%H%M%S")
        shutil.copy2(path, bak)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return bak


def read_conf(conf, nodes_file=""):
    script = r'''. "$1" >/dev/null || exit 1
if [ -n "$2" ] && [ -f "$2" ]; then . "$2" >/dev/null || exit 1; fi
printf 'NODE=%s\n' "${NODES[@]}"
for v in NAME HOST PORT EXTRA_ENV LAUNCHER_PORT LAUNCHER_HOST KV DCP CONTEXT STORE CONTEXT_LENGTH KV_SIZE CACHE RESERVE_GB DRAFTER_KIND DRAFTER ATLAS_DIR ATLAS_GENERIC LAUNCH_DROP_CACHES LAUNCH_SNAPSHOTS RUNTIME IMAGE TOPOLOGY SOCKET_IFNAME IB_HCA WEIGHTS MULTI MULTI_WAIT DRAFT_SAVES; do printf '%s=%s\n' "$v" "${!v:-}"; done'''
    out = subprocess.run(["bash", "-c", script, "_", conf, nodes_file or ""], capture_output=True, text=True, check=True).stdout
    c = {"NODES": []}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "NODE":
            if v:  # NODES=() prints one empty NODE= line
                c["NODES"].append(v)
        else:
            c[k] = v
    return c


def conf_kv(c):
    """(context length, store) as tf-glm53 reads them: CONTEXT_LENGTH, else CONTEXT (819,200); the store KV_SIZE less the
    context length, else STORE (819,200). The engine takes the two (--context, TF_GLM53_PREFIX_STORE)."""
    ctx = int(c.get("CONTEXT_LENGTH") or c.get("CONTEXT") or 819200)
    return ctx, (int(c["KV_SIZE"]) - ctx if c.get("KV_SIZE") else int(c.get("STORE") or 819200))


SETTINGS_SH = r'''keys="$3"
nf=
if [ -n "$2" ] && [ -f "$2" ]; then nf=$(. "$2" >/dev/null 2>&1; for v in $keys; do [ -n "${!v+x}" ] && printf '%s ' "$v"; done); fi
. "$1" >/dev/null || exit 1
if [ -n "$2" ] && [ -f "$2" ]; then . "$2" >/dev/null || exit 1; fi
printf 'N\t%s\n' "$nf"
for v in $keys; do
  if [ "$v" = NODES ]; then [ -n "${NODES+x}" ] && printf 'V\tNODES\t%s\n' "${NODES[*]}"
  elif [ -n "${!v+x}" ]; then printf 'V\t%s\t%s\n' "$v" "${!v}"; fi
done'''


def read_settings(conf, nodes_file=""):
    """SETTINGS' keys (and SET_HIDDEN's) as tf-glm53 reads them: the conf, then the node file. {KEY: (value, set,
    in_node_file)}: value None when unset (NODES: the addresses joined by a space). A clean environment (PATH and HOME
    only), so a variable of the launcher's own never reads as set."""
    keys = " ".join(k for k in list(SETTINGS) + list(SET_HIDDEN) if k not in SET_CANON)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~")}
    out = subprocess.run(["bash", "-c", SETTINGS_SH, "_", conf, nodes_file or "", keys], capture_output=True, text=True,
                         check=True, env=env).stdout
    nf, got = set(), {}
    for line in out.splitlines():
        f = line.split("\t", 2)
        if f[0] == "N" and len(f) > 1:
            nf = set(f[1].split())
        elif f[0] == "V" and len(f) == 3:
            got[f[1]] = f[2]
    return {k: (got.get(k), k in got, k in nf) for k in keys.split()}


def prompt_conf(k, v):
    """A value as glm53.conf holds it: the prompts percent-encoded, empty when the default."""
    if k not in SET_PROMPTS:
        return v
    return "" if v == SETTINGS[k][0] else quote(v, safe="")


def rule_why(rule, v):
    """Why `v` breaks `rule` (a SETTINGS rule), "" when it holds."""
    t = rule["type"]
    if t == "int":
        if v == "" and rule.get("empty"):
            return ""
        if not re.fullmatch(r"0|[1-9][0-9]*", v):  # no leading zeros: bash's $(( )) reads 08 as octal
            return "a whole number" + (" (or empty)" if rule.get("empty") else "")
        n = int(v)
        if n == 0 and rule.get("zero"):
            return ""
        if not rule["min"] <= n <= rule["max"] or n % rule.get("step", 1):
            return (f"0, or {rule['min']:,} to {rule['max']:,}" if rule.get("zero") else f"{rule['min']:,} to {rule['max']:,}") + \
                (f" in steps of {rule['step']:,}" if rule.get("step") else "")
        return ""
    if t == "enum":
        return "" if v in rule["options"] else "one of " + ", ".join(rule["options"])
    if t == "bool":
        return "" if v in ("0", "1") else "0 or 1"
    if len(v) > rule.get("max", 255) or (rule.get("pattern") and not re.fullmatch(rule["pattern"], v)):
        return "not in the form this key takes"
    return ""


class Fail(Exception):
    pass


def run_state(snap):
    """A run snapshot's state: running (not ended or not decided yet), saved, pending, discarded; the first launchers
    kept `folded` instead."""
    if "state" in snap:
        return snap["state"]
    return "saved" if snap.get("folded") else "running"


def no_take():
    """What a start took from saved allocations, when it took nothing: alloc (the allocation it started), its banks by
    width, usage_from (the allocation whose usage it took over) and that usage (base)."""
    return {"alloc": "", "bank_ids": {}, "usage_from": "", "base": None}


def add_u64(x, y):
    """Two u64 LE base64 arrays of [layers x experts] added (an empty one counts as zeros)."""
    a = struct.unpack(f"<{L * E}Q", base64.b64decode(x)) if x else (0,) * (L * E)
    b = struct.unpack(f"<{L * E}Q", base64.b64decode(y)) if y else (0,) * (L * E)
    return base64.b64encode(struct.pack(f"<{L * E}Q", *(p + q for p, q in zip(a, b)))).decode()


def usage_norm(u, saved_unix=0):
    """A saved allocation's usage as it is kept now: rows {decode, prompt}, total {decode, prompt?} (u64 LE base64),
    runs, since_unix (the first run's start), saved_unix. A usage saved before runs and since_unix counts as one run
    from its started_unix. None when there is none or its totals do not hold."""
    if not isinstance(u, dict) or not isinstance(u.get("total"), dict) or not u["total"].get("decode"):
        return None
    try:
        total = {k: u["total"][k] for k in ("decode", "prompt") if u["total"].get(k)}
        if any(len(base64.b64decode(v, validate=True)) != 8 * L * E for v in total.values()):
            return None
        rows = u.get("rows") or {}
        return {"rows": {"decode": int(rows.get("decode", 0)), "prompt": int(rows.get("prompt", 0))}, "total": total,
                "runs": int(u.get("runs", 1)), "since_unix": int(u.get("since_unix", u.get("started_unix", 0))),
                "saved_unix": int(u.get("saved_unix", saved_unix))}
    except (ValueError, TypeError, AttributeError):
        return None


class Launcher:
    def __init__(self, a):
        self.conf = os.path.abspath(a.conf)
        self.pkg_local = os.path.dirname(self.conf)
        self.pkg = a.pkg or self.pkg_local  # the lane directory on the nodes
        self.dry = a.dry_run
        # the lane's nodes as the panel's node check wrote them (glm53.nodes.conf), read after CONF: its lines win
        self.node_file = nodes_file_of(self.conf)
        c = read_conf(self.conf, self.node_file)
        self.nodes = c["NODES"]
        self.world = len(self.nodes)
        self.apply_conf(c)
        # the run's usage snapshot, the pending runs and what the running start took over: in memory only (written to
        # disk when a run is saved: usage-total.json, an allocation); a launcher started again forgets them
        self.mem = {}
        self.weights = None  # R-018: weights_info of WEIGHTS and DRAFTER, read once (weights_view)
        self.drafter_state = {}
        # the nodes' check (POST /v1/launch/check): the last result, one at a time
        self.check = {"state": "idle", "started_unix": 0, "ended_unix": 0, "nodes": [], "ok": False, "why": "", "workers": [],
                      "ring": {"state": "na", "why": ""}, "deployed": False, "head": "", "topology": None, "nodes_ok": False}
        # the weights' locations checked on every node (POST /v1/launch/weights/check): the last result, one at a time
        self.wcheck = {"state": "idle", "started_unix": 0, "ended_unix": 0, "asked": None, "nodes": [], "items": [], "ok": False, "why": "",
                       "found": []}
        self.wcheck_lock = threading.Lock()
        self.check_lock = threading.Lock()
        env = dict(t.split("=", 1) for t in shlex.split(c.get("EXTRA_ENV", "")) if "=" in t)
        self.page = env.get("TF_PANEL_HTML", os.path.join(self.pkg_local, "panel.html"))
        self.sets = env.get("TF_GLM53_EXPERT_SETS", os.path.join(self.pkg_local, "expert-sets"))
        self.listen = (a.host or c.get("LAUNCHER_HOST") or "0.0.0.0", a.port or int(c.get("LAUNCHER_PORT") or self.engine[1] + 10))
        self.dir = os.path.join(self.pkg_local, "launch")
        os.makedirs(self.dir, exist_ok=True)
        self.settings_path = env.get("TF_PANEL_SETTINGS", os.path.join(self.dir, "panel-settings.json"))
        self.settings_lock = threading.Lock()
        # the banks found on the nodes: {bits: {"id", "ok", "nodes": {name: "ok" | why}}} and when (bank_check)
        self.bank_state, self.bank_t, self.bank_busy = {}, 0.0, False
        self.bank_lock = threading.Lock()
        self.progress = os.path.join(self.pkg, "launch", "load-progress.json")
        self.banks = os.path.join(self.pkg_local, "expert-plan", "banks.json")
        self.lock = threading.Lock()
        self.op_lock = threading.Lock()  # one start being admitted at a time
        self.conf_lock = threading.Lock()  # one save of the engine settings at a time (R-024)
        # what the start took from saved allocations (no_take: alloc, bank_ids, usage_from, base)
        self.st = dict({"state": "stopped", "phase": "", "t0": 0.0, "bits": None, "set": "", "error": "", "seen": {}, "restart": False}, **no_take())
        self.job = None  # the start or stop thread
        self.proc = None  # tf-glm53 engine-up / engine-down
        self.cancel = False
        self.engine_up = False
        self.node_info = []
        self.node_t = 0.0
        self.names = list(self.nodes)
        self.engine_bits = None
        self.start_before = {}  # each node's MemAvailable before the current start
        self.usage_lock = threading.RLock()  # keep_usage holds it around end_run
        self.completions = None  # the engine's completion_tokens_total at the last /health
        self.serving = False  # a request runs or waits (the last /health)
        # Expert Atlas's own directory, apart from the lane's: saved allocations in allocs/{builtin,user}
        self.atlas = c.get("ATLAS_DIR") or os.path.join(self.pkg_local, "atlas")
        self.alloc_lock = threading.Lock()  # one save, rename or delete of an allocation at a time
        # the hosts over ConnectX-7, the ring, the lanes' traffic and bandwidth (R-017)
        self.servers = Servers(self, c.get("SOCKET_IFNAME") or "", (c.get("IB_HCA") or "").replace(",", " ").split())

    def apply_conf(self, c):
        """What the launcher keeps from glm53.conf (and the node file) besides the nodes, from read_conf's `c`: read at
        the start, and again once the conf changed while the engine is stopped (conf_refresh; the engine settings'
        save, R-024). The nodes themselves: reload_nodes."""
        port = int(c.get("PORT") or 8000)
        host = c.get("HOST") or "127.0.0.1"
        # how tf-glm53 runs the ranks: from the lane directory, or each in a container (RUNTIME=docker: the lane's
        # containers are glm53-p<PORT>-r<rank>, their servers' host process ids the engine's)
        self.runtime = c.get("RUNTIME") or "host"
        # full: a switch or one link; ring4: four nodes cabled as a ring (glm53/release/RING.md), four nodes only
        self.topology = c.get("TOPOLOGY") or "full"
        self.cname = f"glm53-p{port}-r"
        self.engine = ("127.0.0.1" if host in ("0.0.0.0",) or host.startswith("127.") or host == "localhost" else host, port)
        # the logs: in memory, in the lane's log keeper on each node (tools/logkeep.py; tf-glm53 passes the command)
        import pwd
        self.keep = os.environ.get("TF_LOGKEEP") or f"python3 {self.pkg}/tools/logkeep.py --sock tf-glm53-{pwd.getpwuid(os.getuid()).pw_name}-p{port}"
        self.kv_format = c.get("KV") or "fp8"
        self.dcp = int(c.get("DCP") or (3 if self.world == 3 else 2))  # tf-glm53's default
        # the positions CONF asks for: a request's context and the conversation store beside it; a start in a format
        # that cannot hold them all gets a shorter context (kv_choice)
        self.context, self.store = conf_kv(c)
        self.cache = c.get("CACHE") or "pool"  # tf-glm53's default
        # what a node keeps for the OS after a start (a plan may ask another: reserve_gib)
        self.reserve = int(float(c.get("RESERVE_GB") or RESERVE / GiB) * GiB)
        self.kv_positions_now = self.context + self.store
        # the drafter CONF names (a start may take another: drafter) and DFlash2's model, checked on every node with
        # the banks (drafter_state)
        self.drafter_conf = c.get("DRAFTER_KIND") or "mtp"
        self.drafter_path = c.get("DRAFTER") or ""
        # the conversations decoded together and their wait for room (a start may take others: multi, multi_wait), and
        # DFlash2's kept drafter states (MULTI with DFlash2 needs one at least): tf-glm53's defaults when CONF has none
        self.multi_conf = int(c.get("MULTI") or 0)
        self.multi_wait_conf = int(c.get("MULTI_WAIT") or MULTI_WAIT_DEFAULT)
        self.draft_saves = int(c.get("DRAFT_SAVES") or 8)
        self.weights_dir = c.get("WEIGHTS") or ""
        env = dict(t.split("=", 1) for t in shlex.split(c.get("EXTRA_ENV", "")) if "=" in t)
        # what a live swap keeps to spare on every rank (the engine's TF_GLM53_SWAP_MARGIN_GIB, 2 by default)
        try:
            self.swap_margin = int(env.get("TF_GLM53_SWAP_MARGIN_GIB", "2")) * GiB
        except ValueError:
            self.swap_margin = 2 * GiB
        # opt-in steps of a start or a stop: the page cache dropped (sudo), the nodes' state written down (snapshot)
        self.drop_caches = (c.get("LAUNCH_DROP_CACHES") or "0") == "1"
        self.snapshots = (c.get("LAUNCH_SNAPSHOTS") or "0") == "1"
        # every layer on the generic grouped kernel, so a re-sort may move any expert
        self.generic = (c.get("ATLAS_GENERIC") or "1") == "1"
        self.conf_seen = self.conf_mtimes()

    def conf_mtimes(self):
        """glm53.conf's and the node file's mtimes (0 for one that is not there): what conf_refresh compares."""
        def mt(p):
            try:
                return os.path.getmtime(p) if p else 0
            except OSError:
                return 0
        return mt(self.conf), mt(self.node_file)

    def own_job(self):
        """Nothing else runs: no start or stop thread, or the thread asking is that thread (a restart's)."""
        return not (self.job and self.job.is_alive() and self.job is not threading.current_thread())

    def conf_refresh(self, force=False):
        """The conf read again when it changed since it was last read (a save of the engine settings while the engine
        ran, an edit by hand), only while the engine is stopped: a running engine's views keep the conf it started
        with. Called when a plan, a start or the status is asked for, and after a stop."""
        with self.lock:
            idle = self.st["state"] in ("stopped", "failed") and self.own_job() and not self.engine_up
        if idle and (force or self.conf_mtimes() != self.conf_seen):
            self.reload_conf()

    def reload_conf(self):
        """glm53.conf and the node file read again: the conf's values (apply_conf), the nodes when they changed
        (reload_nodes), the weights' and the drafter's checks when WEIGHTS or DRAFTER changed."""
        try:
            c = read_conf(self.conf, self.node_file)
        except subprocess.CalledProcessError as e:
            log(f"{os.path.basename(self.conf)} does not read: {(e.stderr or '').strip()[-200:]}")
            self.conf_seen = self.conf_mtimes()
            return
        old = (self.weights_dir, self.drafter_path)
        nodes_before = (list(self.nodes), self.topology, self.servers.ifname, list(self.servers.hcas))
        self.apply_conf(c)
        nodes_now = (c["NODES"], c.get("TOPOLOGY") or "full", c.get("SOCKET_IFNAME") or "", (c.get("IB_HCA") or "").replace(",", " ").split())
        if nodes_now != nodes_before:
            self.reload_nodes()
        elif (self.weights_dir, self.drafter_path) != old:
            self.weights = None
            with self.bank_lock:
                self.bank_state, self.bank_t, self.drafter_state = {}, 0.0, {}
            threading.Thread(target=self.bank_check, kwargs={"force": True}, daemon=True).start()
        self.conf_seen = self.conf_mtimes()

    def ssh(self, node, cmd, timeout=20):
        if node.startswith(EXAMPLE_ADDR):  # no node is there: not 8 s of waiting on every read until the node check
            return 255, ""
        try:
            r = subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", node, cmd],
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            return 255, ""

    def probe(self, rank):
        b = os.path.join(self.pkg, "bin", "tensorfold-native")
        mine_cmd = (f"for p in $(pgrep -x tensorfold-nati); do e=$(readlink /proc/$p/exe); [ \"${{e% (deleted)}}\" = {b} ] && echo engine $p; done; "
                    if self.runtime != "docker" else
                    f"for c in $(docker ps -q -f name='^/{self.cname}[0-9]+$'); do docker top $c -eo pid,comm | awk '$2 ~ /^tensorfold/ {{print \"engine\", $1}}'; done; ")
        cmd = ("awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print \"mem\", t*1024, a*1024}' /proc/meminfo; " + mine_cmd +
               "if g=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1); then echo \"$g\" | sed 's/^/gpu /'; "
               "else echo \"gpuerr $(echo \"$g\" | head -1)\"; fi; "
               # the GPU's temperature, and the hottest thermal zone (on GB10 the SoC's ACPI zones) in millidegrees
               "echo gputemp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); "
               "echo cputemp $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1); "
               "echo host $(hostname)")
        rc, out = self.ssh(self.nodes[rank], cmd)
        n = {"name": self.names[rank], "rank": rank, "mem_total": 0, "mem_available": 0, "engine": False, "other": "", "gpu_error": "", "reachable": rc == 0,
             "gpu_temp_c": None, "cpu_temp_c": None}
        mine, gpu = set(), []
        for line in out.splitlines():
            f = line.split(None, 1)
            if not f:
                continue
            if f[0] == "mem" and len(f) == 2:
                t, av = f[1].split()
                n["mem_total"], n["mem_available"] = int(float(t)), int(float(av))
            elif f[0] == "engine":
                mine.add(f[1].strip())
            elif f[0] == "gpu" and len(f) == 2:
                gpu.append([x.strip() for x in f[1].split(",", 1)])
            elif f[0] == "gpuerr":                 # nvidia-smi failed: its first line, not a GPU process
                n["gpu_error"] = f[1].strip() if len(f) == 2 else "nvidia-smi failed"
            elif f[0] == "host" and len(f) == 2:
                n["name"] = self.names[rank] = f[1].strip()
            elif f[0] in ("gputemp", "cputemp") and len(f) == 2:   # unreadable: left null
                try:
                    v = float(f[1].strip())
                    if f[0] == "gputemp":
                        n["gpu_temp_c"] = v
                    else:
                        n["cpu_temp_c"] = round(v / 1000, 1)
                except ValueError:
                    pass
        n["engine"] = bool(mine)
        n["pids"] = sorted(mine)
        n["other"] = ", ".join(sorted({os.path.basename(g[1]) if len(g) > 1 else g[0] for g in gpu if g[0] not in mine}))
        return n

    def probe_all(self):
        asked = self.nodes
        with ThreadPoolExecutor(max(1, self.world)) as ex:  # max: a lane with no NODES yet (the node check first)
            info = list(ex.map(self.probe, range(self.world)))
        with self.lock:
            # NODES saved meanwhile: this read is of the old nodes, 8 s late when they do not answer (the example's
            # addresses before the first node check). It must not replace the new nodes': it reads as no memory
            if asked is self.nodes:
                self.node_info, self.node_t = info, time.time()
        return info

    def on_all(self, cmd, timeout=120):
        with ThreadPoolExecutor(max(1, self.world)) as ex:
            return list(ex.map(lambda n: self.ssh(n, cmd, timeout), self.nodes))

    def engine_post(self, path, body, timeout=10):
        c = http.client.HTTPConnection(*self.engine, timeout=timeout)
        try:
            c.request("POST", path, body, {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def thinking_apply(self, want):
        """A settings save's live keys (SET_LIVE) sent to the running engine (POST /v1/thinking): the keys it took, and
        why not ("" when it took them)."""
        body = {f: (want[k] if f in SET_LIVE_TEXT else int(want[k])) for k, f in SET_LIVE.items() if k in want}
        if not body:
            return [], ""
        try:
            st, raw = self.engine_post("/v1/thinking", json.dumps(body).encode())
        except OSError as e:
            return [], f"the engine does not answer ({e})"
        if st != 200:
            try:
                err = json.loads(raw or b"{}").get("error")
                why = err.get("message", "") if isinstance(err, dict) else str(err or "")
            except ValueError:
                why = ""
            return [], f"{st}" + (f" {why}" if why else "")
        return sorted(k for k in want if k in SET_LIVE), ""

    def thinking_post(self, raw):
        """POST /v1/thinking through the launcher (the inference screen's [ THINKING ]): the engine's answer, and on its
        200 the change written to glm53.conf's lines too (a share also THINKING_BUDGET=0, as the engine sets it), so the
        settings screen and the next start have it. No copy aside for these one-line writes."""
        try:
            st, body = self.engine_post("/v1/thinking", raw or b"{}")
        except OSError as e:
            return 503, {"error": f"the engine does not answer ({e})"}
        try:
            ans = json.loads(body or b"{}")
        except ValueError:
            return 502, {"error": "the engine's answer did not parse"}
        if st == 200 and isinstance(ans, dict):
            try:
                asked = json.loads(raw or b"{}")
            except ValueError:
                asked = {}
            lines = {k: str(ans[f]) for k, f in SET_LIVE.items() if isinstance(asked, dict) and f in asked and f in ans}
            if "THINKING_SHARE" in lines:
                lines["THINKING_BUDGET"] = "0"
            if lines:
                with self.conf_lock:
                    text = open(self.conf).read()
                    new = text
                    for k, v in lines.items():
                        new = set_line(new, k, prompt_conf(k, v))
                    if new != text:
                        write_whole(self.conf, new, keep=False)
                log(f"thinking changed on the running engine and in {os.path.basename(self.conf)}: {' '.join(f'{k}={v}' for k, v in lines.items())}")
        return st, ans

    def engine_get(self, path, timeout=3):
        c = http.client.HTTPConnection(*self.engine, timeout=timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def health(self):
        """Whether the engine answers /health; its completion_tokens_total is kept (self.completions) so the usage is
        read again once the engine has decoded, and whether a request runs or waits (self.serving)."""
        try:
            st, body = self.engine_get("/health", 2)
            if st != 200:
                return False
            try:
                h = json.loads(body)
                self.completions = h.get("completion_tokens_total", self.completions)
                self.serving = bool(h.get("requests_running") or (h.get("live") or {}).get("waiting"))
            except ValueError:
                pass
            return True
        except OSError:
            return False

    def run_id(self):
        """This run of the engine: rank 0's process id (a new start, a new id)."""
        with self.lock:
            info = list(self.node_info)
        return ",".join(info[0].get("pids", [])) if info else ""

    def keep_usage(self):
        """The running engine's decode picks since its start (and its prompt picks, from an engine that counts them),
        kept as this run's snapshot (last-usage.json). A snapshot of another run still unresolved (the engine went away
        without a choice) goes to the pending runs first."""
        try:
            s, body = self.engine_get("/v1/experts/usage?window=0", 10)
            if s != 200:
                return
            u = json.loads(body)
        except (OSError, ValueError) as e:
            log(f"usage not kept: {e}")
            return
        # the read, the compare and the write under the usage lock, with end_run's; a run already saved, dropped or
        # pending is never written back as running (a stop's end_run may come between the engine's answer and here)
        with self.usage_lock:
            self.keep_usage_locked(u)

    def keep_usage_locked(self, u):
        try:
            run = self.run_id()
            last = self.read_json("last-usage.json")
            if last and last.get("run") == run and run_state(last) != "running":
                return
            if last and run_state(last) == "running" and last.get("run") != run:
                self.end_run(last, None)
            same = (last or {}).get("run") == run
            keep = {"run": run, "state": "running", "layers": u["layers"], "experts": u["experts"], "rows": {"decode": u["rows"]["decode"]},
                    "total": {"decode": u["total"]["decode"]}, "bits": u.get("bits", ""), "saved_unix": int(time.time()),
                    "started_unix": last.get("started_unix", int(time.time())) if same else int(time.time())}
            if "prompt" in u["rows"] and "prompt" in u["total"]:
                keep["rows"]["prompt"] = u["rows"]["prompt"]
                keep["total"]["prompt"] = u["total"]["prompt"]
            self.write_json("last-usage.json", keep)
            self.engine_bits = base64.b64decode(keep["bits"]) if keep["bits"] else self.engine_bits
        except (OSError, ValueError, KeyError) as e:
            log(f"usage not kept: {e}")

    def add_to(self, name, snap, runs=1, prompt_runs=None, mark=None):
        """`snap`'s picks added to the sum in `name` (usage-total.json: the learning data; pending.json: the runs not
        decided yet). The prompt picks kept beside the decode ones (prompt, prompt_rows, prompt_runs, prompt_since_unix)
        when `snap` has them: runs from before the engine counted prompts add none, and are not counted as 0.
        `mark` (a run's key): kept in the sum's `added` (the last 64) in the same write, and a run already marked there is
        not added again (a crash between this write and last-usage.json's adds no run twice)."""
        start = snap.get("started_unix", snap["saved_unix"])
        tot = self.read_json(name) or {"runs": 0, "since_unix": start, "rows": 0, "decode": base64.b64encode(bytes(8 * L * E)).decode()}
        if mark is not None and mark in tot.get("added", []):
            return tot
        if mark is not None:
            tot["added"] = (tot.get("added", []) + [mark])[-64:]
        add = lambda x, y: base64.b64encode(struct.pack(f"<{L * E}Q", *(p + q for p, q in zip(
            struct.unpack(f"<{L * E}Q", base64.b64decode(x)), struct.unpack(f"<{L * E}Q", base64.b64decode(y)))))).decode()
        tot.update(runs=tot["runs"] + runs, rows=tot["rows"] + snap["rows"]["decode"], bits=snap.get("bits", "") or tot.get("bits", ""),
                   saved_unix=snap["saved_unix"], until_unix=max(tot.get("until_unix", 0), snap["saved_unix"]),
                   since_unix=min(tot["since_unix"], start), decode=add(tot["decode"], snap["total"]["decode"]))
        if snap["total"].get("prompt"):
            zero = base64.b64encode(bytes(8 * L * E)).decode()
            tot.update(prompt=add(tot.get("prompt") or zero, snap["total"]["prompt"]), prompt_rows=tot.get("prompt_rows", 0) + snap["rows"].get("prompt", 0),
                       prompt_runs=tot.get("prompt_runs", 0) + (runs if prompt_runs is None else prompt_runs),
                       prompt_since_unix=min(tot.get("prompt_since_unix", start), snap.get("prompt_since_unix", start)))
        self.write_json(name, tot)
        return tot

    def end_run(self, snap=None, save=None):
        """A run that ended: save True adds it to the learning data, False drops it, None keeps it pending (the stop
        asked nothing: a stop elsewhere, a crash). Once a run."""
        with self.usage_lock:
            snap = snap or self.read_json("last-usage.json")
            if not snap or run_state(snap) != "running":
                return
            mark = f"{snap.get('run')}@{snap.get('started_unix', 0)}"  # the run's own key
            if save is True:
                tot = self.add_to("usage-total.json", snap, mark=mark)
                snap["state"] = "saved"
                log(f"usage: run {snap.get('run')} ({snap['rows']['decode']} decode rows) saved: {tot['runs']} runs, {tot['rows']} rows")
            elif save is False:
                snap["state"] = "discarded"
                log(f"usage: run {snap.get('run')} ({snap['rows']['decode']} decode rows) discarded")
            else:
                tot = self.add_to("pending.json", snap, mark=mark)
                snap["state"] = "pending"
                log(f"usage: run {snap.get('run')} ({snap['rows']['decode']} decode rows) pending ({tot['runs']} runs, {tot['rows']} rows)")
            self.write_json("last-usage.json", snap)

    def pending_decide(self, save):
        """POST /v1/launch/usage/save (save) or /discard: the pending runs into the learning data, or away."""
        with self.usage_lock:
            pend = self.read_json("pending.json")
            if not pend:
                return 404, {"error": "NoPending"}
            if save:
                snap = {"total": {"decode": pend["decode"]}, "rows": {"decode": pend["rows"]}, "saved_unix": pend["saved_unix"],
                        "started_unix": pend["since_unix"], "bits": pend.get("bits", "")}
                if pend.get("prompt"):
                    snap["total"]["prompt"] = pend["prompt"]
                    snap["rows"]["prompt"] = pend.get("prompt_rows", 0)
                    snap["prompt_since_unix"] = pend.get("prompt_since_unix", pend["since_unix"])
                tot = self.add_to("usage-total.json", snap, runs=pend["runs"], prompt_runs=pend.get("prompt_runs", 0),
                                  mark=f"pending@{pend['since_unix']}-{pend['saved_unix']}")
                log(f"usage: {pend['runs']} pending runs ({pend['rows']} rows) saved: {tot['runs']} runs, {tot['rows']} rows")
            else:
                log(f"usage: {pend['runs']} pending runs ({pend['rows']} rows) discarded")
            self.mem.pop("pending.json", None)
        return 200, {"pending": None, "saved": bool(save)}

    def usage(self):
        """GET /v1/launch/usage: the learning data (the saved runs' decode picks, and the prompt picks of the runs that
        carried them: prompt_runs, prompt_since_unix), the pending runs, the last run's bits, and the last run itself
        (last: its id, what became of it, when it started and was last read; null before any run)."""
        tot = self.read_json("usage-total.json") or {}
        snap = self.read_json("last-usage.json") or {}
        pend = self.read_json("pending.json")
        zero = base64.b64encode(bytes(8 * L * E)).decode()
        return {"layers": [FIRST, FIRST + L - 1], "experts": E, "rows": {"decode": tot.get("rows", 0), "prompt": tot.get("prompt_rows", 0)},
                "total": {"decode": tot.get("decode") or zero, "prompt": tot.get("prompt") or zero},
                "bits": snap.get("bits") or tot.get("bits", ""), "saved_unix": tot.get("saved_unix", 0), "runs": tot.get("runs", 0),
                "since_unix": tot.get("since_unix", 0), "prompt_runs": tot.get("prompt_runs", 0), "prompt_since_unix": tot.get("prompt_since_unix", 0),
                "pending": {"rows": pend["rows"], "runs": pend["runs"], "since_unix": pend["since_unix"], "until_unix": pend.get("until_unix", pend["saved_unix"]),
                            "prompt_rows": pend.get("prompt_rows", 0), "prompt_runs": pend.get("prompt_runs", 0)} if pend else None,
                "last": {"run": snap.get("run", ""), "state": run_state(snap), "started_unix": int(snap.get("started_unix", snap.get("saved_unix", 0))),
                         "saved_unix": int(snap.get("saved_unix", 0))} if snap.get("total") else None}

    def learning(self, current=None, source="decode", saved=True):
        """The picks a expert of `source` ("decode" or "prompt"): the learning data's (the saved runs) unless `saved` is
        false, plus `current` (the running engine's /v1/experts/usage); the rows, and the saved runs that carried them."""
        tot = (self.read_json("usage-total.json") or {}) if saved else {}
        key, rkey, nkey = ("decode", "rows", "runs") if source == "decode" else ("prompt", "prompt_rows", "prompt_runs")
        counts = list(struct.unpack(f"<{L * E}Q", base64.b64decode(tot[key]))) if tot.get(key) else [0] * (L * E)
        rows, runs = tot.get(rkey, 0), tot.get(nkey, 0)
        if current and current["total"].get(source):
            counts = [x + y for x, y in zip(counts, struct.unpack(f"<{L * E}Q", base64.b64decode(current["total"][source])))]
            rows = rows + current["rows"].get(source, 0)
        return counts, rows, runs

    def write_json(self, name, obj):
        if name in MEMORY_ONLY:
            self.mem[name] = json.loads(json.dumps(obj))
            return
        p = os.path.join(self.dir, name)
        tmp = f"{p}.tmp{os.getpid()}.{threading.get_ident()}"  # one a writer
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, p)

    def settings_get(self):
        """The document the page saved last: {"settings": {...}, "saved_unix": int}; {} and 0 before the first save, and
        when the file does not parse (that file set aside as .bad-<time> for a look, the next save replaces it)."""
        with self.settings_lock:
            try:
                raw = open(self.settings_path, "rb").read()
            except OSError:
                return {"settings": {}, "saved_unix": 0}
            try:
                d = json.loads(raw)
                if not (isinstance(d, dict) and d.get("format") == "glm53-panel-settings" and isinstance(d.get("settings"), dict)):
                    raise ValueError("not a glm53-panel-settings document")
                return {"settings": d["settings"], "saved_unix": int(d.get("saved_unix") or 0)}
            except (ValueError, TypeError) as e:
                bad = f"{self.settings_path}.bad-{time.strftime('%Y%m%d-%H%M%S')}"
                try:
                    os.replace(self.settings_path, bad)
                    log(f"the panel's settings did not parse ({e}): set aside as {bad}")
                except OSError:
                    pass
                return {"settings": {}, "saved_unix": 0}

    def settings_post(self, raw):
        """Replaces the whole document (the page owns its keys; last write wins), written to a temporary file in the same
        directory and renamed."""
        if len(raw) > SETTINGS_MAX:
            return 400, {"error": "the settings are over 64 KiB"}
        try:
            j = json.loads(raw or b"null")
        except ValueError:
            return 400, {"error": "bad JSON"}
        if not isinstance(j, dict) or not isinstance(j.get("settings"), dict):
            return 400, {"error": "a JSON object with settings: an object"}
        now = int(time.time())
        doc = {"format": "glm53-panel-settings", "version": 1, "saved_unix": now, "settings": j["settings"]}
        tmp = f"{self.settings_path}.tmp{os.getpid()}.{threading.get_ident()}"
        with self.settings_lock:
            try:
                with open(tmp, "w") as f:
                    json.dump(doc, f, ensure_ascii=False)
                os.replace(tmp, self.settings_path)
            except OSError as e:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return 500, {"error": f"the settings could not be written: {e}"}
        return 200, {"saved_unix": now}

    def read_json(self, name):
        if name in MEMORY_ONLY:
            return json.loads(json.dumps(self.mem[name])) if name in self.mem else None
        try:
            return json.load(open(os.path.join(self.dir, name)))
        except (OSError, ValueError):
            return None

    def read_progress(self):
        if self.dry:
            return None
        try:
            p = json.load(open(self.progress))
            return p if p.get("format") == "glm53-load-progress" else None
        except (OSError, ValueError):
            return None

    def watch(self):
        """Every second: the engine's health; the state follows a start or a stop made outside the launcher."""
        down, kept_at = 0, None
        while True:
            up = self.health()
            down = 0 if up else down + 1
            with self.lock:
                self.engine_up = up
                busy = self.job is not None and self.job.is_alive()
                s = self.st
                procs = any(n["engine"] for n in self.node_info)
                fresh = time.time() - self.node_t < 8
                if not busy:
                    if up and s["state"] != "running":
                        s.update(state="running", phase="", error="", t0=s["t0"] or time.time(), **no_take())
                        log("the engine is up (not started by the launcher)")
                    elif down >= 3 and s["state"] == "running":
                        s.update(state="stopping" if procs else "failed", phase="", t0=time.time(),
                                 error="" if procs else "the engine stopped (not by the launcher)")
                        log(f"the engine does not answer: {s['error'] or 'its processes are still there'}")
                    elif not up and s["state"] in ("starting", "stopping") and fresh and not procs:
                        s.update(state="stopped", phase="", bits=None, set="", **no_take())
                    elif not up and s["state"] in ("stopped", "failed") and fresh and procs:
                        s.update(state="starting", phase="", t0=time.time(), error="", **no_take())
            with self.lock:
                every = bool(self.node_info) and all(n["reachable"] for n in self.node_info)
            if not up and not busy and not procs and fresh and every:
                snap = self.read_json("last-usage.json")
                if snap and run_state(snap) == "running":
                    self.end_run(snap, None)  # ended without a choice (outside the launcher, a crash): pending
            # the run's usage written once the engine has decoded and no request runs or waits (once a reply, not
            # while it is written; nothing while idle): a stop elsewhere or a crash loses the reply under way at most
            moved = self.completions is not None and self.completions != kept_at
            if up and (self.engine_bits is None or (moved and not self.serving)):
                c = self.completions
                self.keep_usage()  # also the bits of an engine the launcher did not start
                kept_at = c
            time.sleep(1)

    def poll_nodes(self):
        while True:
            try:
                self.probe_all()
            except Exception as e:  # keep polling
                log(f"nodes: {e}")
            time.sleep(5)

    def status(self):
        self.conf_refresh()
        with self.lock:
            s = dict(self.st)
            seen = dict(self.st["seen"])
        done, phase, rb, tb = bytes(L), "", 0, 0
        if s["state"] == "running":
            done = bytes([1]) * L
        elif s["state"] == "starting":
            # checking: the launcher's own pre-flight; then the engine's progress file (none yet while the ranks set up
            # CUDA and NCCL: weights); ready from tf-glm53 while its smoke test runs
            phase = s["phase"] or "weights"
            p = self.read_progress() if phase != "checking" else None
            if p:
                done = base64.b64decode(p["done"]) if p.get("done") else done
                rb, tb = p.get("read_bytes", 0), p.get("total_bytes", 0)
                if phase != "ready":
                    phase = p["phase"]
        bits = s["bits"] if s["bits"] is not None else (self.engine_bits if s["state"] == "running" else None)
        el = int((time.time() - s["t0"]) * 1000) if s["t0"] and s["state"] in ("starting", "stopping") else 0
        return {"state": s["state"], "phase": phase, "layers": L,
                "done": base64.b64encode(done).decode(), "experts_done": "", "experts_loaded": sum(done) * E,
                "layers_done": sum(done), "read_bytes": rb, "total_bytes": tb, "elapsed_ms": el,
                "eta_ms": self.eta(phase, sum(done), seen) if s["state"] == "starting" else 0,
                "bits": base64.b64encode(bits).decode() if bits else "", "set": s["set"],
                "alloc": s.get("alloc", "") if s["state"] != "stopped" else "",
                "usage_from": s.get("usage_from", "") if s["state"] != "stopped" else "", "error": s["error"],
                # R-024: a restart (POST /v1/launch/restart) runs: from the stop until its start is admitted or refused
                "restart": bool(s.get("restart")),
                # the KV the start took: this launcher's start, else the last measured one (a launcher restarted while
                # the engine runs), else CONF's; reserve_gib CONF's where neither recorded one (a start before 5f8491f6)
                "kv": {"reserve_gib": self.reserve / GiB, **(s.get("kv") or (self.read_json("last-start.json") or {}).get("kv") or {"format": self.kv_format, "context": self.context, "store": self.store, "dcp": self.dcp})},
                # the widths whose banks every node has (the launch screen offers only those)
                "banks": self.bank_view(),
                # the drafters a start may take, and the one this start or the last measured one took
                "drafter": dict(self.drafter_view(), running=s.get("drafter") or (self.read_json("last-start.json") or {}).get("drafter", self.drafter_conf)),
                # the conversations decoded together: CONF's, the most, why each N is refused, and this start's
                "multi": self.multi_view(s),
                **({"weights": w} if (w := self.weights_view()) else {})}

    def eta(self, phase, layers_done, seen):
        last = self.read_json("last-start.json")
        if not last or phase not in PHASES:
            return 0
        d = last.get("phases", {})
        if not all(k in d for k in PHASES):
            return 0
        i = PHASES.index(phase)
        left = sum(d[k] for k in PHASES[i + 1:])
        spent = time.time() - seen.get(phase, time.time())
        if phase == "weights":
            left += d["weights"] * (1 - layers_done / L)
        else:
            left += max(0.0, d[phase] - spent)
        return int(left * 1000)

    def kv_layers(self, drafter):
        """The cache layers a start holds: an MTP drafter's layer has its cache too (approximate: its indexer not
        counted)."""
        return KV_LAYERS + (1 if drafter == "mtp" else 0)

    def kv_choice(self, fmt, room, drafter=None, context=None, store=None, slope=0, p0=0):
        """The context and store a start in `fmt` takes with `room` bytes for the KV caches on the tightest node: the
        asked context (CONF's when None) when it fits with the store, else the context cut to what fits (a multiple of
        16,384, the store kept); None when not even 65,536 tokens of context fit. The store: CONF's (None), the one
        asked, or "max": everything the context leaves (a multiple of 16,384)."""
        want = context or self.context
        keep = self.store_floor() if store == "max" else (self.store if store is None else store)
        ctx = self.max_context(fmt, room, drafter, keep, slope, p0)
        if not ctx:
            return None
        ctx = min(ctx, want)
        if store == "max":
            return ctx, self.store_max(self.max_positions(fmt, room, drafter, slope, p0), ctx)
        return ctx, keep

    def store_floor(self):
        """The least store a start may take: the pool cache (CACHE=pool) holds two blocks of 256 beyond the context
        (POOL_MIN_STORE; the engine refuses fewer with PoolTooSmall), the slot cache none."""
        return POOL_MIN_STORE if self.cache == "pool" else 0

    def store_max(self, most, ctx):
        """The store "max" takes: the positions `most` leaves after the context, a multiple of 16,384, at least the
        floor (0 when not even that is left)."""
        left, floor = most - ctx, self.store_floor()
        return max(left // CONTEXT_STEP * CONTEXT_STEP, floor) if left >= floor else 0

    def max_positions(self, fmt, room, drafter=None, slope=0, p0=0):
        """The positions `room` holds in `fmt`; with a measured work slope (bytes a position, the room counted with the
        work at p0 positions) the work's growth beyond p0 takes its share."""
        return int((room + slope * p0) // (self.position_bytes(drafter)[fmt] + slope))

    def work_slope(self, last):
        """The work a position more (bytes; the steepest node's) between the fewest and the most positions of the
        measured starts with this drafter (the record's history), when they are at least 65,536 apart, and the positions
        the record's own start had: (0, p0) until two starts say so (10-28, DFlash2 on the ring: 15.5 GiB at
        426K positions, 16.99 at 557K, 17.3 at 606K)."""
        kv = (last or {}).get("kv") or {}
        p0 = int(kv.get("context") or 0) + int(kv.get("store") or 0)
        hist = [h for h in (last or {}).get("history") or [] if h.get("positions")]
        if len(hist) < 2:
            return 0, p0
        lo, hi = min(hist, key=lambda h: h["positions"]), max(hist, key=lambda h: h["positions"])
        dp = hi["positions"] - lo["positions"]
        names = set(lo["work"]) & set(hi["work"])
        if dp < CONTEXT_MIN or not names:
            return 0, p0
        sl = max((hi["work"][n] - lo["work"][n]) / dp for n in names)
        return max(0, min(sl, WORK_SLOPE_MAX)), p0

    def max_context(self, fmt, room, drafter=None, store=None, slope=0, p0=0):
        """The longest context that fits `room` with `store` (CONF's when None) in `fmt` (a multiple of 16,384, at most
        the engine's limit; 0 below 65,536); `slope`, `p0`: the work's growth (work_slope)."""
        most = self.max_positions(fmt, room, drafter, slope, p0)
        ctx = min(CONTEXT_LIMIT, (most - (self.store if store is None else store)) // CONTEXT_STEP * CONTEXT_STEP)
        return ctx if ctx >= CONTEXT_MIN else 0

    def position_bytes(self, drafter=None):
        layers = self.kv_layers(drafter or self.drafter_conf)
        return {f: (layers * lat + KV_INDEX_LAYERS * KV_INDEX) // self.dcp for f, lat in KV_LATENT.items()}

    def last_start(self, drafter):
        """The last measured start with `drafter` (launch/last-start-<drafter>.json; a last-start.json from before the
        drafters were told apart counts as CONF's drafter's)."""
        last = self.read_json(f"last-start-{drafter}.json")
        if last is None:
            last = self.read_json("last-start.json")
            if last is not None and last.get("drafter", self.drafter_conf) != drafter:
                last = None
        return last

    def weights_view(self):
        """R-018: the checkpoint's bits beside the routed experts (weights_info), read once; None when it does not say."""
        if self.weights is None:
            self.weights = weights_info(self.weights_dir, self.drafter_path) or {}
        return self.weights or None

    def drafter_view(self):
        """The drafters a start may take: MTP and none always, DFlash2 when DRAFTER holds its model on every node (the
        banks' check); with why not, and CONF's."""
        with self.bank_lock:
            st = dict(self.drafter_state)
        why = {}
        if not self.drafter_path:
            why["dflash2"] = "DRAFTER is not set in glm53.conf"
        elif not st:
            why["dflash2"] = "not checked yet"
        elif any(v != "ok" for v in st.values()):
            why["dflash2"] = "; ".join(f"{n}: {v}" for n, v in st.items() if v != "ok")
        return {"conf": self.drafter_conf, "available": [d for d in DRAFTERS if d not in why], "why": why}

    def multi_max_lane(self):
        """The most conversations this lane decodes together (status.multi.max_lane): 1 on three nodes (MULTI is not
        offered there), MULTI_RING_MAX on ring4 (until a ring test passes), else MULTI_MAX."""
        if self.world == 3:
            return 1
        return MULTI_RING_MAX if self.topology == "ring4" else MULTI_MAX

    def multi_lane_why(self, n):
        """Why this lane refuses MULTI=n ("" when it does not), the drafter apart (multi_why): tf-glm53's own refusals,
        checked here first so the screen can grey a choice before a start."""
        if n < 1:
            return ""
        if self.cache != "pool":
            return f"CACHE={self.cache} (needs CACHE=pool)"
        if self.world == 3:
            return "3 nodes"
        if self.dcp != 2:
            return f"DCP={self.dcp} (needs DCP=2)"
        if n > self.multi_max_lane():
            return f"not tested on the ring ({MULTI_RING_MAX} at most)"
        return ""

    def multi_drafter_why(self, drafter):
        """Why `drafter` refuses MULTI >= 1 ("" when it does not): DFlash2 needs a kept drafter state at least."""
        return "DRAFT_SAVES=0" if drafter == "dflash2" and self.draft_saves < 1 else ""

    def multi_why(self, n, drafter):
        """Why a start with MULTI=n and `drafter` is refused ("" when it is not): the lane's reason, else the drafter's."""
        return self.multi_lane_why(n) or (self.multi_drafter_why(drafter) if n >= 1 else "")

    def multi_bytes(self, n, drafter):
        """What MULTI=n holds a node beside the start's other sizes: (idle, load, estimate). MULTI_BY_N where measured
        by the conversations; else idle MULTI_IDLE from 1,
        load MULTI_LOAD from 2 (four nodes' for a node count not measured; estimate when the table has no measured
        value for this node count and drafter)."""
        by_n = MULTI_BY_N.get((self.world, drafter))
        if by_n and n >= 1:
            idle, load = by_n[min(n, max(by_n))]
            return int(idle * GiB), int(load * GiB), False
        idle = MULTI_IDLE if n >= 1 else 0
        if n < 2:
            return idle, 0, False
        load = int(MULTI_LOAD.get(self.world, MULTI_LOAD[4])[drafter] * GiB)
        return idle, load, (self.world, drafter) not in MULTI_MEASURED

    def multi_view(self, s):
        """status.multi: CONF's MULTI and MULTI_WAIT, the engine's most, the lane's most, why each N is refused (the
        lane's, then the drafters'), and what this start took (else the last measured start, else CONF's)."""
        ran = s.get("multi") or (self.read_json("last-start.json") or {}).get("multi") or {"n": self.multi_conf, "wait": self.multi_wait_conf}
        why_drafter = {d: w for d in DRAFTERS if (w := self.multi_drafter_why(d))}
        return {"conf": self.multi_conf, "wait_conf": self.multi_wait_conf, "max": MULTI_MAX, "max_lane": self.multi_max_lane(),
                "why": {str(n): w for n in range(1, MULTI_MAX + 1) if (w := self.multi_lane_why(n))}, "why_drafter": why_drafter,
                "running": int(ran.get("n", 0)), "wait": int(ran.get("wait", self.multi_wait_conf))}

    def plan(self, bits, nodes=None, kv=None, drafter=None, context=None, store=None, reserve_gib=None, multi=None, multi_wait=None):
        """The plan of a start with `bits`, the KV format `kv` and the drafter `drafter` (CONF's when None): each node's
        memory with the KV caches the format's context and store take, the verdict, and both formats' positions (the
        launch screen picks one). `context`: the context asked (CONF's when None), cut when it does not fit; `store`: the
        store asked (CONF's when None, "max": the rest); `reserve_gib`: what a node keeps for the OS (CONF's when None);
        `multi`, `multi_wait`: the conversations decoded together and their wait (CONF's when None), whose memory
        (multi_bytes) a node holds before the KV."""
        fmt = kv or self.kv_format
        drafter = drafter or self.drafter_conf
        mn = self.multi_conf if multi is None else multi
        mw = self.multi_wait_conf if multi_wait is None else multi_wait
        m_idle, m_load, m_est = self.multi_bytes(mn, drafter)
        m_why = self.multi_why(mn, drafter)
        default = defaults(self.world, drafter)
        reserve = int(reserve_gib * GiB) if reserve_gib is not None else self.reserve
        nodes = nodes or self.node_info or self.probe_all()
        last = self.last_start(drafter)
        # the store taking the rest on an estimate keeps ESTIMATE_SLACK more (the sizes not yet measured)
        slack = (ESTIMATE_SLACK if last is None or last.get("world") != self.world else MEASURED_SLACK) if store == "max" else 0
        estimate = last is None or last.get("world") != self.world
        # the work grows with the positions (measured at two contexts at least): taken in by the positions a format holds
        slope, p0 = self.work_slope(last) if not estimate else (0, 0)
        ex = experts_bytes(bits, self.world)
        with self.lock:
            busy = self.engine_up or self.st["state"] in ("starting", "running", "stopping")
            before = dict(self.start_before)
        out, order = [], ["fits", "tight", "no"]
        worst = "fits"
        for n in nodes:
            m = (last or {}).get("nodes", {}).get(n["name"]) if not estimate else None
            sizes = dict(default) if m is None else {k: m[k] for k in default}
            avail = n["mem_available"]
            # available = with this lane's engine stopped: while it starts, runs or stops (or its processes are
            # there), MemAvailable now already lacks what it holds, so take the start's own reading from before it
            if busy or n["engine"]:
                if n["name"] in before:
                    avail = before[n["name"]]
                elif m:
                    avail = m["avail_before"]
                else:
                    cur = self.engine_bits
                    avail += (experts_bytes(cur, self.world) if cur else ex) + sum(default.values())
                    estimate = True
            # what the KV caches could take with these bits: the node less everything but the KV (the conversations
            # decoded together, idle and under load, included), less the reserve
            room = max(0, avail - ex - sizes["other_weights"] - sizes["work"] - m_idle - m_load - reserve - slack)
            out.append({"name": n["name"], "experts": ex, "other_weights": sizes["other_weights"], "work": sizes["work"],
                        "multi": m_idle + m_load, "available": avail, "kv_room": room})
        room = min((n["kv_room"] for n in out), default=0)
        pb = self.position_bytes(drafter)
        # each format: the positions the start would allocate (CONF's, or a shorter context) and their bytes a node
        formats = {}
        for f in pb:
            ch = self.kv_choice(f, room, drafter, context, store, slope, p0)
            ctx, st = ch if ch else (0, self.store if store in (None, "max") else store)
            keep = 0 if store == "max" else (self.store if store is None else store)
            most = self.max_positions(f, room, drafter, slope, p0)
            formats[f] = {"context": ctx, "store": st, "positions": ctx + st if ch else 0, "bytes": (ctx + st) * pb[f] if ch else 0,
                          "max_positions": most, "max_context": self.max_context(f, room, drafter, keep, slope, p0),
                          "max_store": self.store_max(most, ctx) if ch else 0,
                          "cut": bool(ch) and ctx < (context or self.context), "fits": ch is not None}
        kvb = formats[fmt]["bytes"]
        for n in out:
            if slope and formats[fmt]["positions"]:
                n["work"] = max(0, n["work"] + int(slope * (formats[fmt]["positions"] - p0)))
            n["kv"] = kvb
            n["total"] = n["experts"] + n["other_weights"] + kvb + n["work"] + n["multi"]
            n["margin"] = n["available"] - n["total"]
            v = "fits" if n["margin"] >= reserve else "tight" if n["margin"] >= reserve // 2 else "no"
            if not formats[fmt]["fits"]:
                v = "no"
            n["verdict"] = v
            worst = max(worst, v, key=order.index)
        kv = {"format": fmt, "conf_format": self.kv_format, "dcp": self.dcp, "positions_now": formats[fmt]["positions"],
              "context": formats[fmt]["context"], "store": formats[fmt]["store"], "room": room, "position_bytes": pb,
              "positions": {f: x["max_positions"] for f, x in formats.items()}, "formats": formats,
              "conf_context": self.context, "context_asked": context, "context_limit": CONTEXT_LIMIT,
              "conf_store": self.store, "store_asked": store}
        miss = self.missing_bits(set(bits) | {REQUIRED_BITS})
        if miss:
            worst = "no"
        dv = self.drafter_view()
        if drafter not in dv["available"]:
            worst = "no"
        if m_why:
            worst = "no"
        # positions_each: the length each of N conversations holds when all N are in the pool at once
        multi_doc = {"n": mn, "wait": mw, "idle": m_idle, "load": m_load, "estimate": m_est,
                     "positions_each": formats[fmt]["positions"] // mn if mn >= 2 else 0, "why": m_why}
        return {"verdict": worst, "reserve": reserve, "conf_reserve": self.reserve, "reserve_gib": reserve / GiB,
                "estimate": bool(estimate), "nodes": out, "kv": kv,
                "missing_bits": {str(w): why for w, why in miss.items()}, "drafter": dict(dv, kind=drafter), "multi": multi_doc}

    def bits_of(self, body):
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        name = ""
        kv = j.get("kv") if isinstance(j, dict) else None  # the request's, also with a set
        drafter = j.get("drafter") if isinstance(j, dict) else None
        context = j.get("context") if isinstance(j, dict) else None
        store = j.get("store") if isinstance(j, dict) else None
        reserve = j.get("reserve_gib") if isinstance(j, dict) else None
        multi = j.get("multi") if isinstance(j, dict) else None
        multi_wait = j.get("multi_wait") if isinstance(j, dict) else None
        if not isinstance(j, dict):
            raise Fail("a JSON object expected")
        if "set" in j:
            name = str(j["set"])
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", name):
                raise Fail("unknown set")
            try:
                j = json.load(open(os.path.join(self.sets, name + ".json")))
            except (OSError, ValueError):
                raise Fail("unknown set")
        try:
            # no widths named: every expert at 3 bits
            b = base64.b64decode(j["bits"], validate=True) if "bits" in j else bytes([REQUIRED_BITS]) * (L * E)
        except (ValueError, TypeError):
            raise Fail(f"bits: base64 of {L} x {E} bytes, each expert's width (every expert at 4 bit: base64.b64encode(bytes([4]) * {L * E}))")
        if len(b) != L * E or any(v < 2 or v > 5 for v in b):
            raise Fail(f"bits: {L} x {E} bytes ({L * E}, got {len(b)}), each a width of 2 to 5")
        if kv not in (None, "fp4", "fp8"):
            raise Fail("kv: fp4 or fp8")
        if drafter not in (None,) + DRAFTERS:
            raise Fail("drafter: " + ", ".join(DRAFTERS))
        if context is not None:
            if isinstance(context, bool) or not isinstance(context, int) or context < CONTEXT_MIN or context % CONTEXT_STEP:
                raise Fail(f"context: a multiple of {CONTEXT_STEP:,} positions, at least {CONTEXT_MIN:,}")
            context = min(context, CONTEXT_LIMIT)  # the engine's limit (kv.context_limit)
        if store is not None and store != "max":
            if isinstance(store, bool) or not isinstance(store, int) or store < self.store_floor():
                raise Fail(f'store: "max", or a number of tokens ({self.store_floor()} or more' +
                           (", the pool cache's least)" if self.store_floor() else ")"))
        if reserve is not None:
            if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not RESERVE_MIN_GIB <= reserve <= RESERVE_MAX_GIB:
                raise Fail(f"reserve_gib: {RESERVE_MIN_GIB} to {RESERVE_MAX_GIB} GiB")
        if multi is not None and (isinstance(multi, bool) or not isinstance(multi, int) or not 0 <= multi <= MULTI_MAX):
            raise Fail(f"multi: 0 to {MULTI_MAX}")
        if multi_wait is not None and (isinstance(multi_wait, bool) or not isinstance(multi_wait, int) or multi_wait < 0):
            raise Fail("multi_wait: whole seconds, 0 or more")
        return b, name, {"kv": kv, "drafter": drafter, "context": context, "store": store, "reserve_gib": reserve,
                         "multi": multi, "multi_wait": multi_wait}

    def start_take(self, raw):
        """POST /v1/launch/start: what the start takes from saved allocations. "alloc": its bits put in the body's place
        (the bits sent beside it, for a launcher without allocations, are not used), its banks and its usage;
        "usage_from" (without alloc): the bits sent and that allocation's usage (a plan whose shares were changed from an
        opened allocation). The usage is a copy: a rename or a delete later leaves it. (None, body, no_take()) without
        either; (error, body, None) when there is no such allocation (404) or a bank of the alloc is not this lane's (409)."""
        try:
            j = json.loads(raw or b"{}")
        except ValueError:
            return None, raw, no_take()
        if not isinstance(j, dict) or (j.get("alloc") is None and j.get("usage_from") is None):
            return None, raw, no_take()
        take = no_take()
        name = str(j["alloc"] if j.get("alloc") is not None else j["usage_from"])
        a = self.alloc_read(name)
        if a is None:
            return (404, {"error": "NoSuchAlloc"}), raw, None
        if j.get("alloc") is not None:
            problem = self.alloc_problem(a)
            if problem:
                return (409, {"error": problem}), raw, None
            j.pop("set", None)
            j["bits"] = base64.b64encode(a["bits"]).decode()
            take.update(alloc=name, bank_ids={w: a["banks"][w]["id"] for w in a["banks"]})
        j.pop("alloc", None)
        j.pop("usage_from", None)
        take.update(usage_from=name, base=usage_norm(a.get("usage"), a.get("saved_unix", 0)))
        return None, json.dumps(j).encode(), take

    def keep_take(self, take):
        """What a start that runs now took (launch/usage-base.json, with the run's id), so a launcher restarted while
        it runs still knows it (restore_take)."""
        self.write_json("usage-base.json", dict(take, run=self.run_id(), bank_ids={str(w): i for w, i in take["bank_ids"].items()},
                                                saved_unix=int(time.time())))

    def restore_take(self):
        """A launcher started while the engine runs: what that start took, when usage-base.json is this run's."""
        t = self.read_json("usage-base.json")
        if not t or not t.get("run") or t["run"] != self.run_id():
            return
        with self.lock:
            self.st.update(alloc=t.get("alloc", ""), usage_from=t.get("usage_from", ""), base=t.get("base"),
                           bank_ids={int(w): i for w, i in (t.get("bank_ids") or {}).items()})
        if t.get("alloc") or t.get("usage_from"):
            log(f"the running start's allocation: {t.get('alloc') or '-'}, usage from {t.get('usage_from') or '-'}")

    def begin_start(self, bits, name, kv=None, drafter=None, context=None, store=None, reserve_gib=None, take=None, multi=None, multi_wait=None):
        # two starts at once: the second one 409, not a second worker
        if not self.op_lock.acquire(blocking=False):
            return 409, {"error": "a start is being admitted"}
        try:
            return self.begin_start_locked(bits, name, kv, drafter, context, store, reserve_gib, take or no_take(), multi, multi_wait)
        finally:
            self.op_lock.release()

    def begin_start_locked(self, bits, name, kv=None, drafter=None, context=None, store=None, reserve_gib=None, take=None, multi=None, multi_wait=None):
        # (a restart's own thread starts the engine again from here: own_job)
        with self.lock:
            if self.st["state"] not in ("stopped", "failed") or not self.own_job():
                return 409, {"error": f"not stopped ({self.st['state']})"}
        # the conf as it is now (the engine settings saved while the engine ran, an edit by hand), and what this start
        # asked for itself, before the conf's values fill the rest (a restart asks the same again: start-conf.json)
        self.conf_refresh()
        asked = {"kv": kv, "drafter": drafter, "context": context, "store": store, "reserve_gib": reserve_gib, "multi": multi, "multi_wait": multi_wait}
        if self.world not in ENGINE_WORLDS:
            return 412, {"error": f"this engine does not run on {self.world} nodes yet (TP={self.world} needs the three-way engine): "
                                  "plans only, a start is refused"}
        kv = kv or self.kv_format
        drafter = drafter or self.drafter_conf
        # no checkpoint here: refused now, not after the ranks start
        if not os.path.isfile(os.path.join(os.path.expanduser(self.weights_dir), "config.json")):
            return 412, {"error": f"no checkpoint at {self.weights_dir} on this node: put {CHECKPOINT_REPO.get(self.world, 'the checkpoint')} there, the same path on every node "
                                  "(POST /v1/launch/weights/check looks on every node)"}
        # the banks (and DFlash2's model) checked again now: a width without data refused here, not after the ranks start
        miss = self.missing_bits(set(bits) | {REQUIRED_BITS}, force=True)
        if miss:
            return 412, {"error": "no data for " + ", ".join(f"{w} bit ({why})" for w, why in miss.items())
                                  + (f" ({REQUIRED_BITS} bit is always needed)" if REQUIRED_BITS in miss else ""),
                         "missing_bits": {str(w): why for w, why in miss.items()}}
        # every node read now, not the last poll: right after a stop the poll may have caught the memory still being freed
        dv = self.drafter_view()
        if drafter not in dv["available"]:
            return 412, {"error": f"drafter {drafter}: " + dv["why"].get(drafter, "not available"), "drafter": dv}
        # the conversations decoded together: CONF's when not asked; a refused one before the ranks start
        multi = self.multi_conf if multi is None else multi
        multi_wait = self.multi_wait_conf if multi_wait is None else multi_wait
        why = self.multi_why(multi, drafter)
        if why:
            with self.lock:
                s = dict(self.st)
            return 412, {"error": f"multi {multi}: {why}", "multi": self.multi_view(s)}
        p = self.plan(bits, nodes=self.probe_all(), kv=kv, drafter=drafter, context=context, store=store, reserve_gib=reserve_gib,
                      multi=multi, multi_wait=multi_wait)
        if p["verdict"] == "no":
            return 412, {"error": "the plan does not fit", "plan": p}
        take = take or no_take()
        # what this start read of the conf and asked itself (GET /v1/launch/settings pending, POST /v1/launch/restart);
        # its run is added once the engine runs
        self.write_json("start-conf.json", {"unix": int(time.time()), "run": "", "asked": asked,
                                            "conf": {k: x["value"] for k, x in self.es_read().items() if k in SETTINGS}})
        with self.lock:
            self.cancel = False
            self.st.update(state="starting", phase="checking", t0=time.time(), bits=bits, set=name, error="",
                           seen={"checking": time.time()}, **take)
            self.job = threading.Thread(target=self.run_start, args=(bits, name, kv, drafter, context, store, reserve_gib, take, multi, multi_wait), daemon=True)
            self.job.start()
        return 200, {"state": "starting"}

    def set_phase(self, phase):
        with self.lock:
            self.st["phase"] = phase
            self.st["seen"].setdefault(phase, time.time())

    def run_start(self, bits, name, kv, drafter=None, context=None, store=None, reserve_gib=None, take=None, multi=None, multi_wait=None):
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            self.preflight(ts)
            before = {n["name"]: n["mem_available"] for n in self.probe_all()}
            with self.lock:
                self.start_before = dict(before)
            drafter = drafter or self.drafter_conf
            multi = self.multi_conf if multi is None else multi
            multi_wait = self.multi_wait_conf if multi_wait is None else multi_wait
            p = self.plan(bits, kv=kv, drafter=drafter, context=context, store=store, reserve_gib=reserve_gib,
                          multi=multi, multi_wait=multi_wait)
            if p["multi"]["why"]:
                raise Fail(f"multi {multi}: {p['multi']['why']}")
            if p["verdict"] == "no":
                worst = min(p["nodes"], key=lambda n: n["margin"])
                raise Fail(f"the plan does not fit: {worst['name']} margin {worst['margin'] / GiB:.1f} GiB")
            # the KV format the launch screen chose, and the context and store it takes (tf-glm53 reads LAUNCH_KV,
            # LAUNCH_CONTEXT, LAUNCH_STORE over CONF's KV, CONTEXT, STORE)
            # and what a node must have available before it (tf-glm53 check reads LAUNCH_NEED_GB over its own estimate
            # from CONF's STORE): the plan's sizes and the reserve, on the node that needs most
            need = max(n["total"] for n in p["nodes"]) + p["reserve"]
            # and the conversations decoded together (tf-glm53 reads LAUNCH_MULTI, LAUNCH_MULTI_WAIT over CONF's MULTI,
            # MULTI_WAIT); the need above holds their memory
            kvs = {"LAUNCH_KV": kv, "LAUNCH_CONTEXT": str(p["kv"]["context"]), "LAUNCH_STORE": str(p["kv"]["store"]),
                   "LAUNCH_NEED_GB": str(int(need // GiB)), "LAUNCH_MULTI": str(multi), "LAUNCH_MULTI_WAIT": str(multi_wait),
                   # each rank's own need in MiB, rounded up (the nodes' work areas differ: 11-01 a node held to the
                   # largest one's need, in whole GiB, failed the check by less than a GiB)
                   "LAUNCH_NEED_MIB": " ".join(str(-(-(n["total"] + p["reserve"]) // (1 << 20)))
                                               for n in sorted(p["nodes"], key=lambda n: self.names.index(n["name"]) if n["name"] in self.names else 0))}
            with self.lock:
                self.st["kv"] = {"format": kv, "context": p["kv"]["context"], "store": p["kv"]["store"], "dcp": self.dcp,
                                 "reserve_gib": p["reserve"] / GiB}
                self.st["drafter"] = drafter
                self.st["multi"] = {"n": multi, "wait": multi_wait}
            # a saved allocation: the plan maps each width to the bank it was saved with (its id in banks.json, the
            # same manifest: start_take), and the swap banks of those widths are those banks' directories (a swap
            # needs the plan's own bank of a width); the other widths keep the lane's first bank of each
            take = take or no_take()
            ids = take["bank_ids"]
            plan_path = self.write_plan(bits, take["alloc"] or name, ts, ids, f"alloc {take['alloc']}" if take["alloc"] else "tf_launcher")
            # the ranges (the counts by bits) stay as the plan sets them; a re-sort while running moves experts between
            # them through the swap banks, one a width, the plan's own; the generic kernel lets every layer
            # take any width; 5 bits need the prompt kernel's 5-bit path
            dirs = self.bank_dirs(ids)
            # the live counts the panel and a re-sort read (TF_GLM53_EXPERT_LIVE) on every start made here
            env = (f"TF_GLM53_EXPERT_PLAN={plan_path} TF_GLM53_LOAD_PROGRESS={self.progress} TF_GLM53_EXPERT_BANKS={dirs} TF_GLM53_EXPERT_LIVE=1"
                   f" TF_GLM53_MEM_RESERVE_GIB={max(1, int(p['reserve'] // GiB))}")
            if self.generic:
                env += " TF_GLM53_EXPERTS_GENERIC=1"
            if 5 in bits:
                env += " TF_GLM53_PE_K10=1"
            if self.cancel:
                raise Fail("cancelled")
            try:
                os.remove(self.progress)  # the last start's
            except OSError:
                pass
            rc, tail = self.run_tf("up", env, ts, kvs, drafter=drafter)
            if rc != 0:
                why = self.rank_errors()
                raise Fail((why + " / " if why else "") + (tail or f"tf-glm53 engine-up exited {rc}"))
            if self.cancel:  # up finished as the stop came
                self.run_tf("down", "", ts)
                raise Fail("cancelled")
            self.measured(bits, before, drafter)
            # a stop that came while the start measured: down, not running
            with self.lock:
                cancelled = self.cancel
                if not cancelled:
                    self.st.update(state="running", phase="", error="")
            if cancelled:
                self.run_tf("down", "", ts)
                raise Fail("cancelled")
            self.keep_take(take)
            sc = self.read_json("start-conf.json")
            if sc and not sc.get("run"):
                self.write_json("start-conf.json", dict(sc, run=self.run_id()))
            log("running")
        except Fail as e:
            log(f"start failed: {e}")
            with self.lock:
                self.st.update(state="failed" if not self.cancel else "stopped", phase="", error="" if self.cancel else str(e))
        except Exception as e:  # a bug here must not leave the state at starting
            log(f"start failed (launcher): {e!r}")
            with self.lock:
                self.st.update(state="failed", phase="", error=f"launcher: {e!r}")
        finally:
            self.snapshot(ts + "-launch-post-up")

    # ---- the nodes (rank order: the head, then the workers) ----

    def config_view(self):
        """GET /v1/launch/config: NODES in rank order, each node's host name once probed ("" before), the world, when
        glm53.conf was saved, and whether the panel's node check registered the nodes (the node file, its time)."""
        try:
            saved = int(os.path.getmtime(self.conf))
        except OSError:
            saved = 0
        try:
            reg = int(os.path.getmtime(self.node_file)) if self.node_file else 0
        except OSError:
            reg = 0
        with self.lock:
            nodes, names = list(self.nodes), list(self.names)
        return {"nodes": nodes, "names": [n if n != a else "" for n, a in zip(names, nodes)], "world": self.world,
                "worlds": list(self.worlds()), "topology": self.topology, "saved_unix": saved,
                "registered": bool(reg), "registered_unix": reg, "node_file": os.path.basename(self.node_file or ""), "conf_path": os.path.abspath(self.conf)}

    def registered(self):
        """Whether the panel's node check wrote the node file (the next start opens on the launch settings)."""
        return bool(self.node_file) and os.path.isfile(self.node_file)

    def worlds(self):
        """The node counts the panel may set: TESTED_WORLDS, or four alone on a registered ring."""
        return (4,) if self.topology == "ring4" and self.registered() else TESTED_WORLDS

    def own_addrs(self):
        """This node's IPv4 addresses (the head a check or a save names must be one of them)."""
        try:
            out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        return {f[f.index("inet") + 1].split("/")[0] for f in (l.split() for l in out.splitlines()) if "inet" in f}

    def nodes_of(self, body, required):
        """The head and the workers a check or a save names: the head (an address of this node; NODES[0] when the body
        names none) and w - 1 distinct addresses for a node count w of TESTED_WORLDS (the node check judges the
        cabling again, so any tested count), the head not among them (NODES' workers when the body names none and none
        is required)."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        if not isinstance(j, dict):
            raise Fail("bad JSON: an object")
        head = j.get("head")
        if head is None:
            if not self.nodes:
                raise Fail("head: NODES is empty: name the head's address")
            head = self.nodes[0]
        else:
            if not isinstance(head, str) or not ADDR.fullmatch(head.strip()):
                raise Fail("head: an address")
            head = head.strip()
            if head not in self.own_addrs():
                raise Fail(f"head: {head} is not an address of this node")
        w = j.get("workers")
        if w is None and not required:
            return head, list(self.nodes[1:])
        if not isinstance(w, list) or not all(isinstance(x, str) for x in w):
            raise Fail("workers: a list of addresses")
        w = [x.strip() for x in w]
        if len(w) + 1 not in TESTED_WORLDS:
            raise Fail("workers: " + " or ".join(str(n - 1) for n in TESTED_WORLDS) + " addresses (the head stays)")
        bad = [x for x in w if not ADDR.fullmatch(x)]
        if bad:
            raise Fail("workers: not an address: " + ", ".join(bad))
        if len(set(w)) != len(w):
            raise Fail("workers: an address twice")
        if head in w:
            raise Fail("workers: the head is not a worker")
        return head, w

    def stopped(self):
        with self.lock:
            return self.st["state"] in ("stopped", "failed") and not (self.job and self.job.is_alive()) and not self.engine_up

    def check_view(self):
        with self.check_lock:
            return {k: v for k, v in self.check.items()}

    def begin_check(self, body):
        """POST /v1/launch/check: tf-glm53 deploy, then tf-glm53 check, on the head and these workers, in the
        background (the package goes to a new worker first, or its files could not pass), with TOPOLOGY=auto: the
        check judges the cabling (tools/topology.py) and its judgment comes with the answer (topology), the rows in
        the order it ranked the nodes. {"deploy": false}: the check alone, nothing written on a node (the panel's own
        check after a trace); a package not copied yet then fails its row. {"head"}: the address rank 0
        is registered with (one of this node's)."""
        head, workers = self.nodes_of(body, required=False)
        j = json.loads(body or b"{}")
        deploy = j.get("deploy", True)
        if not isinstance(deploy, bool):
            raise Fail("deploy: true or false")
        if not self.stopped():
            return 409, {"error": "the engine is not stopped"}
        with self.check_lock:
            if self.check["state"] == "checking":
                return 409, {"error": "a check is running"}
            self.check = {"state": "checking", "started_unix": int(time.time()), "ended_unix": 0, "nodes": [], "ok": False,
                          "why": "", "workers": workers, "ring": {"state": "na", "why": ""}, "deployed": deploy,
                          "head": head, "topology": None, "nodes_ok": False}
        threading.Thread(target=self.run_check, args=(head, workers, deploy), daemon=True).start()
        return 202, {"state": "checking"}

    def judgment(self, since):
        """The cabling tf-glm53 check judged (TOPOLOGY=auto: PKG/launch/topology-pPORT.json) when written since
        `since`: verdict, mode, order, hca, ifname, why; None when it was not."""
        p = os.path.join(self.pkg_local, "launch", f"topology-p{self.engine[1]}.json")
        try:
            if os.path.getmtime(p) < since:
                return None
            j = json.load(open(p))
        except (OSError, ValueError):
            return None
        return {k: j.get(k) for k in ("verdict", "mode", "order", "hca", "ifname", "why")}

    def run_check(self, head, workers, deploy=True):
        nodes = [head] + workers
        why = []
        ring = {"state": "na", "why": ""}
        t0 = time.time() - 1
        judged = None
        res = []
        # the nodes asked, the cabling judged by the check itself (TOPOLOGY=auto): a node file of this check's own,
        # read after CONF in place of the lane's (NODES_CONF)
        tmp = os.path.join(self.dir, "check-" + time.strftime("%Y%m%d-%H%M%S") + ".nodes.conf")
        try:
            open(tmp, "w").write("# the launcher's check (removed after it)\nTOPOLOGY=auto\nNODES=(" + " ".join(nodes) + ")\n")
            tf = ["bash", os.path.join(self.pkg_local, "tf-glm53"), "-c", self.conf]
            env = dict(os.environ, NODES_CONF=tmp)
            d = subprocess.run(tf + ["deploy"], capture_output=True, text=True, timeout=1800, env=env) if deploy else None
            # CHECK_ITEMS: each node's items as well ("  item<TAB>rank<TAB>key<TAB>state<TAB>value<TAB>fix")
            c = subprocess.run(tf + ["check"], capture_output=True, text=True, timeout=600, env=dict(env, CHECK_ITEMS="1"))
            judged = self.judgment(t0)
            # the rows in the order the check ranked the nodes (a ring: ring order, the head kept)
            order = judged["order"] if judged and isinstance(judged.get("order"), list) and sorted(judged["order"]) == sorted(nodes) else nodes
            res = [{"rank": r, "addr": a, "name": "", "ok": False, "why": "", "detail": "", "items": []} for r, a in enumerate(order)]
            for line in (d.stdout if d else "").splitlines():  # "  FAIL ADDR": the package could not be copied there
                m = re.match(r"\s+FAIL (\S+)$", line)
                if m and m.group(1) in order:
                    res[order.index(m.group(1))]["why"] = "the package could not be copied (tf-glm53 deploy)"
            if "SHA256SUMS does not match here" in (d.stdout if d else c.stdout):
                why.append("this directory does not match its SHA256SUMS")
            for line in c.stdout.splitlines():
                it = re.match(r"\s+item\t(\d+)\t(\w+)\t(ok|warn|fail)\t([^\t]*)\t(.*)$", line)
                if it:
                    if int(it.group(1)) < len(res):
                        res[int(it.group(1))]["items"].append(
                            {"key": it.group(2), "state": it.group(3), "value": it.group(4), "fix": it.group(5)})
                    continue
                rg = re.match(r"\s+ring: (ok|NG)\b:?\s*(.*)$", line)
                if rg:
                    ring = {"state": "ok" if rg.group(1) == "ok" else "fail", "why": "" if rg.group(1) == "ok" else rg.group(2)}
                m = re.match(r"\s+rank (\d+) (\S+): (ok|NG)(?::|\s)\s*(.*)$", line)
                if m and int(m.group(1)) < len(res):
                    r = res[int(m.group(1))]
                    if m.group(3) == "ok":
                        r["ok"], r["detail"] = not r["why"], m.group(4).strip().strip("()")
                    else:
                        r["why"] = ", ".join(x for x in (r["why"], m.group(4).strip()) if x)
                # the check's own lines that say what it found, not a problem (the drafter, the replies, the notes;
                # the ring's line is the answer's ring)
                elif line.strip() and not re.match(r"\s*(drafter:|check: ok|replies:|concurrent:|image:|note:|logs:|ring:|\(DRAFTER is set)", line) and not m:
                    if c.returncode != 0 and not line.startswith("check:"):
                        why.append(line.strip())
            if c.returncode != 0 and not c.stdout.strip() and c.stderr.strip():
                why.append(c.stderr.strip().splitlines()[-1])
            # the banks row from the launcher's own check of the banks, then every node's rows in the panel's order
            for r, b in zip(res, self.bank_items(order)):
                if b and any(i["key"] == "ssh" and i["state"] == "ok" for i in r["items"]):
                    r["items"].append(b)
                r["items"].sort(key=lambda i: CHECK_ORDER.index(i["key"]) if i["key"] in CHECK_ORDER else len(CHECK_ORDER))
            with ThreadPoolExecutor(len(order)) as ex:
                names = list(ex.map(lambda a: self.ssh(a, "hostname", 15)[1].strip(), order))
            for r, n in zip(res, names):
                r["name"] = n
                if not r["ok"] and not r["why"]:
                    r["why"] = "not reached by tf-glm53 check"
            ok = c.returncode == 0 and (d is None or d.returncode == 0) and all(r["ok"] for r in res) and bool(judged and judged.get("mode"))
            if c.returncode == 0 and not (judged and judged.get("mode")):
                why.append("the check judged no cabling (tools/topology.py)")
            # the nodes themselves fit (what a registration needs): every node read, the cabling judged, the ring's
            # check not failed, and nothing failed but where the weights are (set afterwards on the launch settings)
            nodes_ok = (bool(judged and judged.get("mode")) and ring["state"] != "fail" and (d is None or d.returncode == 0)
                        and all(any(i["key"] == "ssh" and i["state"] == "ok" for i in r["items"]) for r in res)
                        and all(i["key"] in NODES_OK_FAILS for r in res for i in r["items"] if i["state"] == "fail")
                        and not any(r["why"] == "the package could not be copied (tf-glm53 deploy)" for r in res))
        except (OSError, subprocess.TimeoutExpired, Fail) as e:
            why.append(str(e))
            ok = nodes_ok = False
            if not res:
                res = [{"rank": r, "addr": a, "name": "", "ok": False, "why": str(e), "detail": "", "items": []} for r, a in enumerate(nodes)]
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        with self.check_lock:
            self.check.update(state="done", ended_unix=int(time.time()), nodes=res, ok=ok, why="; ".join(why[:5]), ring=ring, topology=judged,
                              nodes_ok=nodes_ok)
        log(f"check of {' '.join(nodes)}: {'ok' if ok else 'NG'}{', the nodes ok' if nodes_ok and not ok else ''}" + (f" ({judged['verdict']}, {judged['mode']})" if judged else "")
            + ("" if ok else ": " + "; ".join(f"{r['addr']}: {r['why']}" for r in res if r["why"])))

    def save_config(self, body):
        """POST /v1/launch/config: the node file written (TOPOLOGY, NODES, SOCKET_IFNAME, IB_HCA as the check judged
        them) once a check found the nodes fit (nodes_ok: only where the weights are may have failed, set afterwards on
        the launch settings) on exactly this head and these workers; glm53.conf is not touched. The previous node
        file copied aside first, the new one written whole (a temporary file, then a rename), then everything read
        again."""
        head, workers = self.nodes_of(body, required=True)
        if not self.stopped():
            return 409, {"error": "the engine is not stopped"}
        ch = self.check_view()
        t = ch.get("topology") or {}
        if ch["state"] != "done" or not ch.get("nodes_ok") or ch.get("workers") != workers or ch.get("head") != head or not t.get("mode"):
            return 412, {"error": "check this list first (POST /v1/launch/check), then save it"}
        order = t["order"] if isinstance(t.get("order"), list) and sorted(t["order"]) == sorted([head] + workers) else [head] + workers
        what = {"ring4": "a ring of four", "full": "a switch" if t.get("verdict") == "switch" else "one link" if t.get("verdict") == "direct" else "full"}.get(t["mode"], t["mode"])
        text = (f"# The lane's nodes, written by the control panel's node check on {time.strftime('%Y-%m-%d %H:%M:%S')}: {what}, checked.\n"
                "# tf-glm53 and the launcher read this file after glm53.conf, so these lines win. The panel's node check writes it again.\n"
                f"TOPOLOGY={t['mode']}\nNODES=({' '.join(order)})\n")
        if t.get("ifname"):
            text += f"SOCKET_IFNAME={t['ifname']}\n"
        if t.get("hca"):
            text += f"IB_HCA={t['hca']}\n"
        if not self.node_file:
            return 409, {"error": "NODES_CONF is empty: this launcher reads no node file"}
        keep = write_whole(self.node_file, text)
        self.reload_nodes()
        log(f"nodes registered in {os.path.basename(self.node_file)}: {t['mode']} {' '.join(order)}" + (f" (the previous one in {os.path.basename(keep)})" if keep else ""))
        return 200, self.config_view()

    def reload_nodes(self):
        """glm53.conf and the node file read again: probes, the banks' and the drafter's checks and plans use the new
        nodes; the topology, DCP and the devices the ring and the scans read follow."""
        c = read_conf(self.conf, self.node_file)
        with self.lock:
            self.nodes = c["NODES"]
            self.world = len(self.nodes)
            self.names = list(self.nodes)
            self.node_info, self.node_t = [], 0.0
            self.topology = c.get("TOPOLOGY") or "full"
            self.dcp = int(c.get("DCP") or (3 if self.world == 3 else 2))
        self.servers.ifname = c.get("SOCKET_IFNAME") or ""
        self.servers.hcas = (c.get("IB_HCA") or "").replace(",", " ").split()
        with self.bank_lock:
            self.bank_state, self.bank_t, self.drafter_state = {}, 0.0, {}
        threading.Thread(target=self.bank_check, kwargs={"force": True}, daemon=True).start()
        threading.Thread(target=self.probe_all, daemon=True).start()

    # ---- where the weights are (the launch settings): WEIGHTS, DRAFTER, the expert banks' directories ----
    def conf_raw(self, key):
        """KEY's value as glm53.conf writes it (its last KEY= line, quotes and a trailing comment taken off; ~ kept),
        None when it has no such line."""
        v = None
        try:
            for line in open(self.conf).read().splitlines():
                m = re.match(rf"{key}=(.*)$", line)
                if m:
                    v = re.sub(r"\s+#.*$", "", m.group(1)).strip()
        except OSError:
            return None
        if v is not None and len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        return v

    def banks_raw(self):
        """banks.json's banks as written (~ kept): [{id, bits, dir, manifest?}]."""
        try:
            return [b for b in json.load(open(self.banks)).get("banks", []) if isinstance(b, dict) and "id" in b]
        except (OSError, ValueError, AttributeError):
            return []

    def browse(self, raw):
        """GET /v1/launch/browse?path=P: P's subdirectories on the head (this node) for the launch settings' directory
        dialog, each with what it holds: checkpoint (config.json and model.safetensors.index.json), bank (an expert
        bank's layer files, experts-layer-NNN.safetensors), model (config.json alone). Directories only, hidden ones
        left out, under BROWSE_ROOTS only; a path under the home directory comes back as ~/..., as banks.json writes
        it (the same on every node). No path: ~/models when there is one, else the home directory."""
        home = os.path.realpath(os.path.expanduser("~"))
        roots = []
        for r in BROWSE_ROOTS:
            a = os.path.realpath(os.path.expanduser(r))
            if os.path.isdir(a) and a not in roots:
                roots.append(a)
        shown = lambda a: "~" + a[len(home):] if a == home or a.startswith(home + "/") else a  # noqa: E731
        p = (raw or "").strip() or ("~/models" if os.path.isdir(os.path.expanduser("~/models")) else "~")
        if not WPATH.fullmatch(p) and p != "~":
            return 400, {"error": "path: ~/... or /..., of A-Za-z0-9._@+/-"}
        a = os.path.realpath(os.path.expanduser(p))
        root = next((r for r in roots if a == r or a.startswith(r.rstrip("/") + "/")), None)
        # hidden directories stay out whatever the path says (a .. or a link into one)
        if root is None or any(x.startswith(".") for x in a[len(root):].split("/")):
            return 400, {"error": f"path: only under {', '.join(shown(r) for r in roots)}, no hidden directory"}
        if not os.path.isdir(a):
            return 400, {"error": f"{shown(a)}: no such directory on {socket.gethostname()}"}
        try:
            names = sorted(n for n in os.listdir(a) if not n.startswith("."))
        except OSError as e:
            return 400, {"error": f"{shown(a)}: {e.strerror}"}
        dirs, more = [], False
        for n in names:
            d = os.path.join(a, n)
            if not os.path.isdir(d):
                continue
            if len(dirs) >= BROWSE_MAX:
                more = True
                break
            try:
                inside = set(os.listdir(d))
                kinds = []
                if "config.json" in inside and "model.safetensors.index.json" in inside:
                    kinds.append("checkpoint")
                if any(x.startswith("experts-layer-") and x.endswith(".safetensors") for x in inside):
                    kinds.append("bank")
                if "config.json" in inside and "checkpoint" not in kinds:
                    kinds.append("model")
                if any(x.endswith(".json") for x in inside) and not kinds and os.path.basename(d) == "expert-banks":
                    kinds.append("manifests")
                dirs.append({"name": n, "path": shown(d), "kinds": kinds})
            except OSError:
                dirs.append({"name": n, "path": shown(d), "kinds": [], "unreadable": True})
        return 200, {"host": socket.gethostname(), "path": shown(a), "parent": None if a == root else shown(os.path.dirname(a)),
                     "roots": [shown(r) for r in roots], "dirs": dirs, "truncated": more}

    def weights_get(self):
        """GET /v1/launch/weights: WEIGHTS and DRAFTER as glm53.conf has them, every bank's directory as banks.json
        has it, and the last check of locations (POST /v1/launch/weights/check)."""
        w, d = self.conf_raw("WEIGHTS"), self.conf_raw("DRAFTER")
        try:
            saved = int(os.path.getmtime(self.conf))
        except OSError:
            saved = 0
        with self.wcheck_lock:
            ch = {k: v for k, v in self.wcheck.items() if not k.startswith("_")}
        return {"weights": w if w is not None else self.weights_dir, "drafter": d if d is not None else self.drafter_path,
                "drafter_kind": self.drafter_conf, "saved_unix": saved,
                "banks": [{"id": b["id"], "bits": b.get("bits"), "dir": b.get("dir", ""), "repo": b.get("repo", ""), "required": b.get("bits") == REQUIRED_BITS,
                           "manifest": b.get("manifest") or os.path.join(b.get("dir", ""), b["id"] + ".json")} for b in self.banks_raw()],
                "check": json.loads(json.dumps(ch))}

    def weights_body(self, body):
        """A check's or a save's locations: weights, drafter ("" for none), every bank of banks.json once (id, dir);
        each path ~/... or /..., of A-Za-z0-9._@+/- (it goes into glm53.conf unquoted), at most 255."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        if not isinstance(j, dict):
            raise Fail("bad JSON: an object")
        def path(v, what, empty=False):
            if empty and v == "":
                return ""
            if not isinstance(v, str) or not WPATH.fullmatch(v) or len(v) > 255:
                raise Fail(f"{what}: a path ~/... or /... of letters, digits and ._@+/- (at most 255)")
            return v.rstrip("/") or "/"
        w = path(j.get("weights"), "weights")
        d = path(j.get("drafter", ""), "drafter", empty=True)
        have = [b["id"] for b in self.banks_raw()]
        bs = j.get("banks", [])
        if not isinstance(bs, list) or not all(isinstance(x, dict) for x in bs):
            raise Fail("banks: a list of {id, dir}")
        ids = [x.get("id") for x in bs]
        if sorted(map(str, ids)) != sorted(have) or len(set(ids)) != len(ids):
            raise Fail("banks: every bank of expert-plan/banks.json once (" + ", ".join(have) + ")")
        dirs = {x["id"]: path(x.get("dir", ""), f"banks {x['id']}", empty=True) for x in bs}   # "": that width not used
        return {"weights": w, "drafter": d, "banks": [{"id": i, "dir": dirs[i]} for i in have]}

    def begin_weights_check(self, body):
        """POST /v1/launch/weights/check: the locations checked on every node at once, in the background."""
        asked = self.weights_body(body)
        if not self.nodes:
            raise Fail("NODES is empty: run the node check first")
        with self.wcheck_lock:
            if self.wcheck["state"] == "checking":
                return 409, {"error": "a check of the weights runs"}
            self.wcheck = {"state": "checking", "started_unix": int(time.time()), "ended_unix": 0, "asked": asked, "nodes": [],
                           "items": [], "ok": False, "why": "", "found": [], "_nodes": list(self.nodes)}
        threading.Thread(target=self.run_weights_check, args=(asked,), daemon=True).start()
        return 202, {"state": "checking"}

    def run_weights_check(self, asked):
        raw = {b["id"]: b for b in self.banks_raw()}
        # each row: its width (banks.json's), the directory asked, banks.json's manifest while that directory is its own
        banks = [{"row": b["id"], "bits": raw.get(b["id"], {}).get("bits"), "dir": b["dir"],
                  "bj": raw.get(b["id"], {}).get("manifest") or "" if b["dir"] == raw.get(b["id"], {}).get("dir") else ""} for b in asked["banks"]]
        arg = base64.b64encode(json.dumps({"weights": asked["weights"], "drafter": asked["drafter"], "banks": banks}).encode()).decode()
        code = base64.b64encode(WEIGHTS_PROBE.encode()).decode()
        nodes = list(self.nodes)
        why = []
        try:
            res = self.on_all(f"python3 -c \"$(echo {code} | base64 -d)\" {arg}", 120)
        except Exception as e:
            res, why = [(255, "")] * len(nodes), [repr(e)]
        keys = ["weights"] + (["drafter"] if asked["drafter"] else []) + ["bank:" + b["id"] for b in asked["banks"]]
        rows, infos = [], []
        for r, (a, (rc, o)) in enumerate(zip(nodes, res)):
            try:
                got = json.loads(o.strip().splitlines()[-1])
                name = got.pop("_host", "") or a
            except (ValueError, IndexError):
                got, name = None, a
                why.append(f"{a}: " + ("not reachable over ssh" if rc == 255 else f"the check failed (rc {rc})"))
            items, info = [], {}
            for k in keys:
                val = (got.get(k) or ["fail", "not checked"]) if got is not None else ["fail", "not reachable"]
                st, v = val[0], val[1]
                if len(val) > 2 and val[2]:
                    info[k] = val[2]
                items.append({"key": k, "state": st, "value": v, "fix": ""})
            rows.append({"addr": a, "name": name, "rank": r, "items": items})
            infos.append(info)
        # every node the head's bank of a row (the same manifest, by its sha256)
        head = infos[0] if infos else {}
        for row, info in zip(rows[1:], infos[1:]):
            for it in row["items"]:
                h, mine = head.get(it["key"]), info.get(it["key"])
                if it["key"].startswith("bank:") and it["state"] == "ok" and h and mine and mine["manifest_sha256"] != h["manifest_sha256"]:
                    it.update(state="fail", value=f"a different bank here ({mine.get('id')}, the head's {h.get('id')})")
        for row in rows:
            for it in row["items"]:
                if it["state"] != "ok":
                    bank = next((b for b in banks if "bank:" + b["row"] == it["key"]), None)
                    repo = raw.get(bank["row"], {}).get("repo") if bank else ""
                    it["fix"] = {"weights": f"put {CHECKPOINT_REPO.get(self.world, 'the checkpoint')} at {asked['weights']} on {row['name']}",
                                 "drafter": f"DFlash2's model at {asked['drafter']} on {row['name']}"}.get(
                        it["key"], f"{bank['bits']} bit: put {repo or 'its files'} at {bank['dir']} on {row['name']}" if bank and bank["dir"] else "")
        summary = []
        for k in keys:
            its = [(row["name"], next(i for i in row["items"] if i["key"] == k)) for row in rows]
            sts = {i["state"] for _, i in its}
            if sts == {"ok"}:
                summary.append({"key": k, "state": "ok", "value": its[0][1]["value"] if its else ""})
            elif k.startswith("bank:") and sts == {"warn"} and raw.get(k[5:], {}).get("bits") == REQUIRED_BITS:
                summary.append({"key": k, "state": "fail", "value": f"{REQUIRED_BITS} bit is always needed: on no node",
                                "fix": its[0][1]["fix"].replace(f" on {its[0][0]}", " on every node")})
            elif k.startswith("bank:") and sts == {"ok", "warn"} and raw.get(k[5:], {}).get("bits") != REQUIRED_BITS:
                # another width on some nodes only: not offered, like one on no node. A start that asks for it is refused (missing_bits)
                summary.append({"key": k, "state": "warn", "value": "not on " + ", ".join(n for n, i in its if i["state"] != "ok") + ": that width is not offered",
                                "fix": next((i["fix"] for _, i in its if i["state"] != "ok" and i["fix"]), "")})
            elif k.startswith("bank:") and sts == {"warn"}:
                blank = not next((b["dir"] for b in asked["banks"] if "bank:" + b["id"] == k), "")
                summary.append({"key": k, "state": "warn", "value": "not used" if blank else "on no node: that width is not offered",
                                "fix": "" if blank else its[0][1]["fix"].replace(f" on {its[0][0]}", " on every node")})
            else:
                summary.append({"key": k, "state": "fail", "value": "; ".join(f"{n}: {i['value']}" for n, i in its if i["state"] != "ok"),
                                "fix": "; ".join(i["fix"] for _, i in its if i["state"] != "ok" and i["fix"])})
        # the bank each row found on the head (a save writes banks.json's entry from it)
        found_full = {k[5:]: v for k, v in head.items() if k.startswith("bank:")}
        found = [{"row": b["id"], "bits": found_full[b["id"]].get("bits"), "id": found_full[b["id"]].get("id"),
                  "manifest": found_full[b["id"]].get("manifest"), "manifest_sha256": found_full[b["id"]].get("manifest_sha256"),
                  "how": found_full[b["id"]].get("how")} for b in asked["banks"] if b["id"] in found_full]
        ok = all(x["state"] != "fail" for x in summary)
        with self.wcheck_lock:
            self.wcheck.update(state="done", ended_unix=int(time.time()), nodes=rows, items=summary, ok=ok, why="; ".join(why[:5]),
                               found=found, _found=found_full)
        log(f"weights check: {'ok' if ok else 'NG'}: " + ", ".join(f"{x['key']} {x['state']}" for x in summary))

    def ssh_in(self, node, cmd, data, timeout=30):
        """cmd on node over ssh with `data` on its standard input: (rc, the last line of its output or error)."""
        try:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", node, cmd], input=data,
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            return 255, ["timed out"]

    def weights_save(self, body):
        """POST /v1/launch/weights: once a check of exactly these locations passed on the lane's nodes, glm53.conf's
        WEIGHTS= and DRAFTER= lines and banks.json written (each copied aside first, written whole): a row whose bank
        the check found differs from its entry (another directory, id or manifest) is rewritten from that manifest;
        banks.json copied to every worker (the package's identity: the same on every node), then read again."""
        asked = self.weights_body(body)
        if not self.stopped():
            return 409, {"error": "the engine is not stopped"}
        with self.wcheck_lock:
            wc = dict(self.wcheck)
        if wc["state"] != "done" or not wc["ok"] or wc.get("asked") != asked or wc.get("_nodes") != list(self.nodes):
            return 412, {"error": "check these locations first (POST /v1/launch/weights/check), then save them"}
        text = open(self.conf).read()
        new = set_line(set_line(text, "WEIGHTS", asked["weights"]), "DRAFTER", asked["drafter"])
        kept = []
        if new != text:
            kept.append(write_whole(self.conf, new))
        try:
            doc = json.load(open(self.banks))
        except (OSError, ValueError) as e:
            raise Fail(f"expert-plan/banks.json: {e}")
        dirs = {b["id"]: b["dir"] for b in asked["banks"]}
        found = wc.get("_found") or {}
        changed = False
        out = []
        for b in doc.get("banks", []):
            rid = b.get("id") if isinstance(b, dict) else None
            f = found.get(rid)
            if rid not in dirs:
                out.append(b)
            elif f and (b.get("dir") != dirs[rid] or f.get("id") != rid or f.get("manifest_sha256") != b.get("manifest_sha256")):
                # the bank the check found for this width, written from its manifest (another bank: another id, its
                # own manifest; repo and note kept only for the same bank's id)
                nb = {"id": f["id"], "bits": b.get("bits"), "calibration": f.get("calibration"), "dir": dirs[rid], "manifest": f["manifest"]}
                if f["id"] == rid:
                    nb.update({k: b[k] for k in ("repo", "note") if k in b})
                nb.update(manifest_sha256=f["manifest_sha256"], layers=f.get("layer_count"), first_layer=f.get("first_layer"))
                if f.get("bytes") is not None:
                    nb["bytes"] = f["bytes"]
                nb.update(codebook=f.get("codebook"), stored_ranks=f.get("stored_ranks"))
                out.append({k: v for k, v in nb.items() if v is not None})
                changed = True
            elif b.get("dir") != dirs[rid]:
                # nothing found there (that width not offered): the directory alone, the manifest then <dir>/<id>.json
                b = dict(b, dir=dirs[rid])
                b.pop("manifest", None)
                out.append(b)
                changed = True
            else:
                out.append(b)
        doc["banks"] = out
        if changed:
            kept.append(write_whole(self.banks, json.dumps(doc, indent=1, ensure_ascii=False) + "\n"))
        data = open(self.banks).read()
        dst = shlex.quote(os.path.join(self.pkg, "expert-plan", "banks.json"))
        cmd = f"mkdir -p {shlex.quote(os.path.join(self.pkg, 'expert-plan'))} && cat > {dst}.tmp$$ && mv -f {dst}.tmp$$ {dst} && echo copied"
        with ThreadPoolExecutor(max(1, len(self.nodes) - 1)) as ex:
            got = list(ex.map(lambda n: self.ssh_in(n, cmd, data), self.nodes[1:]))
        copied = [{"addr": n, "ok": rc == 0, "why": "" if rc == 0 else (last[0] or f"rc {rc}")} for n, (rc, last) in zip(self.nodes[1:], got)]
        c = read_conf(self.conf, self.node_file)
        self.weights_dir, self.drafter_path = c.get("WEIGHTS") or "", c.get("DRAFTER") or ""
        self.weights = None
        with self.bank_lock:
            self.bank_state, self.bank_t, self.drafter_state = {}, 0.0, {}
        threading.Thread(target=self.bank_check, kwargs={"force": True}, daemon=True).start()
        self.forget_check_weights()
        log(f"weights saved: WEIGHTS={asked['weights']} DRAFTER={asked['drafter']}" + (", banks.json's directories" if changed else "")
            + (f" (kept {', '.join(os.path.basename(k) for k in kept if k)})" if any(kept) else "")
            + ("; banks.json not copied to " + ", ".join(x["addr"] for x in copied if not x["ok"]) if not all(x["ok"] for x in copied) else ""))
        return 200, dict(self.weights_get(), copied=copied)

    def forget_check_weights(self):
        """The last node check's rows for the weights and the banks were about the paths before this save: gone, or the
        node screen goes on failing a lane whose weights are found (the save itself follows a check of these paths)."""
        with self.check_lock:
            nodes = self.check.get("nodes") or []
            for n in nodes:
                n["items"] = [i for i in n.get("items", []) if i["key"] not in NODES_OK_FAILS]
                if n["items"] and all(i["state"] != "fail" for i in n["items"]):
                    n.update(ok=True, why="")
            if nodes and self.check.get("nodes_ok") and all(n["ok"] for n in nodes):
                self.check.update(ok=True, why="")

    # ---- the engine settings (R-024): glm53.conf's keys read and written from the panel's settings screen ----
    def es_table(self):
        """SETTINGS as this lane takes it: DCP's options by the node count (3 nodes: 1 or 3), LAUNCHER_PORT's default
        PORT + 10. {KEY: (default, rule, when)}."""
        t = dict(SETTINGS)
        if self.world == 3:
            t["DCP"] = ("3", SET_ENUM("1", "3"), "start")
        t["LAUNCHER_PORT"] = (str(self.engine[1] + 10),) + t["LAUNCHER_PORT"][1:]
        return t

    def es_read(self):
        """Each key's effective value (a string; the default when unset), whether the conf, the node file or
        EXTRA_ENV names it, and where from: {KEY: {"value", "set", "from"}} (SET_HIDDEN's too)."""
        t = self.es_table()
        raw = read_settings(self.conf, self.node_file)
        out = {}
        for k, (v, isset, nf) in raw.items():
            d = t[k][0] if k in t else ""
            out[k] = {"value": d if v is None else v, "set": isset, "from": "node_file" if nf else "conf" if isset else "default"}
            if k in SET_PROMPTS:  # empty in the conf: the default
                out[k] = {"value": unquote(v or "") or d, "set": bool(v), "from": "conf" if v else "default"}
        # EXTRA_ENV's tokens: TF_GLM53_CANON_* under their own keys, the other tokens as EXTRA_ENV
        try:
            toks = shlex.split(out["EXTRA_ENV"]["value"])
        except ValueError:
            toks = out["EXTRA_ENV"]["value"].split()
        rest = []
        if out.get("THINKING_FOLD_SHARE", {}).get("value") == "0":   # a conf from before: 0 meant what 100 means now
            out["THINKING_FOLD_SHARE"]["value"] = "100"
        for k in SET_CANON:
            out[k] = {"value": t[k][0], "set": False, "from": "default"}
        for tok in toks:
            name, eq, v = tok.partition("=")
            if eq and name in SET_CANON:
                out[name] = {"value": v, "set": True, "from": "extra_env"}
            else:
                rest.append(tok)
        out["EXTRA_ENV"]["value"] = " ".join(rest)
        # a conf with the old names only: CONTEXT_LENGTH from CONTEXT, KV_SIZE from CONTEXT + STORE (each 819,200 when
        # unset), as tf-glm53 reads them
        old = lambda k: raw[k][0] if raw[k][1] else None
        if not out["CONTEXT_LENGTH"]["set"] and old("CONTEXT") is not None:
            out["CONTEXT_LENGTH"] = {"value": old("CONTEXT"), "set": True, "from": "conf"}
        if not out["KV_SIZE"]["set"] and re.fullmatch(r"[0-9]+", out["CONTEXT_LENGTH"]["value"]):
            st = old("STORE") or "819200"
            if re.fullmatch(r"[0-9]+", st):
                out["KV_SIZE"] = {"value": str(int(out["CONTEXT_LENGTH"]["value"]) + int(st)), "set": raw["STORE"][1] or raw["CONTEXT"][1],
                                  "from": "conf" if raw["STORE"][1] or raw["CONTEXT"][1] else "default"}
        return out

    def es_when(self, k, cur):
        """When a change of `k` is used (SETTINGS' when; TOPOLOGY, SOCKET_IFNAME, IB_HCA read only once the node file
        names them)."""
        w = self.es_table()[k][2]
        return "readonly" if k in SET_NODE_FILE and cur[k]["from"] == "node_file" else w

    def start_conf(self):
        """What the running start read (launch/start-conf.json), when it is this run's: {"conf": {KEY: value},
        "asked": {kv, drafter, ...}}; None for an engine started elsewhere or before the launcher recorded one."""
        sc = self.read_json("start-conf.json")
        run = self.run_id()
        return sc if sc and run and sc.get("run") == run else None

    def es_pending(self, cur):
        """The keys whose value in the conf now differs from what the running start read; [] while stopped, and for an
        engine started outside the launcher (nothing recorded)."""
        with self.lock:
            running = self.st["state"] == "running"
        sc = self.start_conf() if running else None
        if not sc:
            return []
        was = sc.get("conf") or {}
        t = self.es_table()
        return [k for k in t if k in was and was[k] != cur[k]["value"] and t[k][2] != "live"
                and not (k == "THINKING_BUDGET" and cur[k]["value"] == "0")]

    def es_view(self, cur=None):
        """GET /v1/launch/settings."""
        cur = cur or self.es_read()
        t = self.es_table()
        try:
            saved = int(os.path.getmtime(self.conf))
        except OSError:
            saved = 0
        with self.lock:
            state = self.st["state"]
        keys = [{"key": k, "value": cur[k]["value"], "default": d, "set": cur[k]["set"], "from": cur[k]["from"],
                 "when": self.es_when(k, cur), "rule": rule} for k, (d, rule, _) in t.items()]
        return {"conf": os.path.basename(self.conf), "saved_unix": saved, "state": state, "keys": keys,
                "pending": self.es_pending(cur)}

    def es_cross(self, v, changed):
        """The rules across keys on the values after a change (`v`: KEY -> value): {KEY: why}, each rule reported only
        when one of its keys is among `changed` (a conf already against a rule by hand still takes other changes), on
        the first of its keys that changed."""
        errs = {}

        def rule(keys, ok, why):
            hit = [k for k in keys if k in changed]
            if hit and not ok and hit[0] not in errs:
                errs[hit[0]] = why
        num = lambda k: int(v[k]) if re.fullmatch(r"[0-9]+", v.get(k) or "") else 0
        multi = num("MULTI")
        rule(["MULTI", "CACHE"], multi == 0 or v["CACHE"] == "pool", f"MULTI={multi} needs CACHE=pool")
        rule(["MULTI", "DCP"], multi == 0 or v["DCP"] == "2", f"MULTI={multi} needs DCP=2")
        rule(["MULTI"], multi == 0 or self.world != 3, f"MULTI={multi} is not offered on 3 nodes yet")
        rule(["MULTI", "DRAFTER_KIND", "DRAFT_SAVES"], multi == 0 or v["DRAFTER_KIND"] != "dflash2" or num("DRAFT_SAVES") > 0,
             f"MULTI={multi} with DFlash2 needs DRAFT_SAVES of 1 or more")
        rule(["DRAFTER_KIND"], v["DRAFTER_KIND"] != "dflash2" or bool(v["DRAFTER"]),
             "DRAFTER_KIND=dflash2 needs DRAFTER (the DFlash2 model's directory, on the [ WEIGHTS ] card)")
        pool = v["CACHE"] == "pool"
        rule(["KV_SIZE", "CONTEXT_LENGTH", "CACHE"], num("KV_SIZE") >= num("CONTEXT_LENGTH") + (POOL_MIN_STORE if pool else 0),
             f"KV_SIZE must be CONTEXT_LENGTH + {POOL_MIN_STORE:,} or more with CACHE=pool" if pool else "KV_SIZE must be CONTEXT_LENGTH or more")
        rule(["DCP", "CACHE"], v["CACHE"] != "pool" or v["DCP"] in ("1", "2"), "CACHE=pool takes DCP 1 or 2")
        rule(["MASTER_PORT", "PORT"], v["MASTER_PORT"] != v["PORT"], "MASTER_PORT must differ from PORT")
        rule(["LAUNCHER_PORT", "PORT", "MASTER_PORT"], v["LAUNCHER_PORT"] not in (v["PORT"], v["MASTER_PORT"]),
             "LAUNCHER_PORT must differ from PORT and MASTER_PORT")
        fold = v["THINKING_FOLD"] != "off"
        fsh, tsh = num("THINKING_FOLD_SHARE"), num("THINKING_SHARE")
        rule(["THINKING_FOLD_SHARE", "THINKING_SHARE", "THINKING_FOLD"], not fold or fsh == 100 or tsh == 0 or fsh < tsh,
             "THINKING_FOLD_SHARE must be below THINKING_SHARE, or 100 (no earlier than the limit)")
        ring = v["TOPOLOGY"] == "ring4"
        rule(["TOPOLOGY"], not ring or self.world == 4, "TOPOLOGY=ring4 needs 4 nodes")
        rule(["TOPOLOGY", "DCP"], not ring or v["DCP"] in ("1", "2"), "TOPOLOGY=ring4 takes DCP 1 or 2")
        names = [t.partition("=")[0] for t in (v["EXTRA_ENV"] or "").split()]
        rule(["EXTRA_ENV"], "TF_GLM53_MULTI" not in names, "set MULTI instead of TF_GLM53_MULTI in EXTRA_ENV")
        rule(["EXTRA_ENV"], not any(n in SET_CANON for n in names), "TF_GLM53_CANON_CLOSE, TF_GLM53_CANON_NEAR and TF_GLM53_BYTE_RESUME have keys of their own")
        bad = [n for n in names if n in RING_ENV]
        rule(["EXTRA_ENV", "TOPOLOGY"], not ring or not bad, f"TOPOLOGY=ring4: {', '.join(bad)} in EXTRA_ENV would change what the ring needs")
        return errs

    def es_post(self, body):
        """POST /v1/launch/settings {"values": {KEY: "value"}}: each changed key checked by its rule and across keys,
        then glm53.conf's lines written (set_line; the conf copied aside once, written whole). While stopped the
        launcher reads the conf again at once; while running when the engine stops (and at every start)."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        vals = j.get("values") if isinstance(j, dict) else None
        if not isinstance(vals, dict) or not vals:
            raise Fail('values: an object of KEY: "value"')
        with self.conf_lock:
            cur = self.es_read()
            t = self.es_table()
            errs, want = {}, {}
            for k, v in vals.items():
                if k not in t:
                    errs[k] = "not a key of the engine settings"
                elif isinstance(v, bool) or not isinstance(v, (str, int)):
                    errs[k] = "a string"
                elif (str(v).strip() or (t[k][0] if k in SET_PROMPTS else "")) == cur[k]["value"]:
                    continue
                elif self.es_when(k, cur) == "readonly":
                    errs[k] = ("read only here: the node check writes it" if k in SET_NODE_FILE or k == "NODES"
                               else "read only here: set it on the launch settings' [ WEIGHTS ] card")
                else:
                    v = str(v).strip() or (t[k][0] if k in SET_PROMPTS else "")
                    why = rule_why(t[k][1], v)
                    if why:
                        errs[k] = why
                    else:
                        want[k] = v
            if not errs and want:
                errs = self.es_cross(dict({k: x["value"] for k, x in cur.items()}, **want), set(want))
            if errs:
                return 400, {"error": "; ".join(f"{k}: {why}" for k, why in errs.items()), "errors": errs}
            busy = [k for k in want if t[k][2] == "stopped"]
            if busy and not self.stopped():
                return 409, {"error": f"{', '.join(busy)}: only while the engine is stopped", "errors": {k: "only while the engine is stopped" for k in busy}}
            kept = ""
            if want:
                text = open(self.conf).read()
                new = text
                for k, v in want.items():
                    if k not in SET_CANON and k != "EXTRA_ENV" and k not in ("CONTEXT_LENGTH", "KV_SIZE"):
                        new = set_line(new, k, prompt_conf(k, v))
                if "THINKING_SHARE" in want and cur["THINKING_BUDGET"]["value"] != "0":
                    new = set_line(new, "THINKING_BUDGET", "0")   # the share decides, as on the engine
                if "THINKING_FOLD_CAP_SHARE" in want:
                    new = drop_lines(new, ("THINKING_FOLD_CAP",))   # the percent decides
                if "MAX_TOKENS_SHARE" in want and "MAX_TOKENS" not in want and cur["MAX_TOKENS"]["value"]:
                    new = set_line(new, "MAX_TOKENS", "")   # the share decides
                if "CONTEXT_LENGTH" in want or "KV_SIZE" in want:
                    # both written (KV_SIZE kept as it read, so a conf of the old names keeps its whole KV) and the old
                    # names' lines dropped
                    for k in ("CONTEXT_LENGTH", "KV_SIZE"):
                        new = set_line(new, k, want.get(k, cur[k]["value"]))
                    new = drop_lines(new, SET_OLD)
                if "EXTRA_ENV" in want or any(k in want for k in SET_CANON):
                    # TF_GLM53_CANON_* kept as tokens while they differ from the default or were already there unchanged
                    toks = (want["EXTRA_ENV"] if "EXTRA_ENV" in want else cur["EXTRA_ENV"]["value"]).split()
                    for k in SET_CANON:
                        v = want.get(k, cur[k]["value"])
                        if v != t[k][0] or (k not in want and cur[k]["set"]):
                            toks.append(f"{k}={v}")
                    new = set_line(new, "EXTRA_ENV", '"' + " ".join(toks) + '"')
                if new != text:
                    kept = write_whole(self.conf, new)
                    try:
                        read_settings(self.conf, self.node_file)
                    except subprocess.CalledProcessError as e:  # never leave a conf tf-glm53 cannot read
                        if kept:
                            shutil.copy2(kept, self.conf)
                        return 500, {"error": f"the written {os.path.basename(self.conf)} did not read ({(e.stderr or '').strip()[-200:]}): the previous one put back"}
                log(f"engine settings saved: {' '.join(f'{k}={v}' for k, v in want.items())}" + (f" (the previous conf in {os.path.basename(kept)})" if kept else ""))
            cur = self.es_read()
        # the live keys sent to a running engine at once (R-029)
        with self.lock:
            running = self.st["state"] == "running"
        applied, why = self.thinking_apply(want) if running else ([], "")
        self.conf_refresh()
        out = dict(self.es_view(cur), changed=sorted(want), kept=os.path.basename(kept) if kept else "", applied=applied)
        if why:
            out["apply_error"] = why
        return 200, out

    def begin_restart(self, body):
        """POST /v1/launch/restart {"alloc"?: name}: the engine stopped as a stop with save_usage false, then started
        again with the bits it runs (with alloc: that saved allocation, as a start's alloc) and the running start's own
        choices, except those whose conf key was saved since (es_pending): the conf's then."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        if not isinstance(j, dict):
            raise Fail("a JSON object expected")
        alloc = j.get("alloc")
        if alloc is not None and (not isinstance(alloc, str) or not ALLOC_NAME.fullmatch(alloc)):
            raise Fail("alloc: an allocation's name")
        if alloc is not None:
            a = self.alloc_read(alloc)
            if a is None:
                return 404, {"error": "NoSuchAlloc"}
            problem = self.alloc_problem(a)
            if problem:
                return 409, {"error": problem}
        cur = self.es_read()
        pend = set(self.es_pending(cur))
        sc = self.start_conf() or {}
        asked = {f: (None if key in pend else (sc.get("asked") or {}).get(f)) for f, key in SET_ASKED.items()}
        with self.lock:
            s = self.st
            if s["state"] != "running" or (self.job and self.job.is_alive()):
                return 409, {"error": f"not running ({s['state']})" if s["state"] != "running" else f"busy ({s['state']})"}
            bits = s["bits"] if s["bits"] is not None else self.engine_bits
            if bits is None:
                return 409, {"error": "the bits loaded are not read yet"}
            take = {k: s.get(k) for k in no_take()}
            name = s.get("set", "")
            s.update(state="stopping", phase="", t0=time.time(), restart=True)
            self.job = threading.Thread(target=self.run_restart, args=(bits, name, take, asked, alloc), daemon=True)
            self.job.start()
        log("restart: stopping, then starting with " + (f"allocation {alloc}" if alloc else "the bits loaded") +
            (f"; from the conf: {', '.join(sorted(SET_ASKED[f] for f in SET_ASKED if SET_ASKED[f] in pend))}" if pend & set(SET_ASKED.values()) else ""))
        return 202, {"state": "stopping"}

    def run_restart(self, bits, name, take, asked, alloc):
        try:
            self.run_stop(False)
            with self.lock:
                stopped = self.st["state"] == "stopped"
            if not stopped:
                return
            if alloc is not None:
                a = self.alloc_read(alloc)
                problem = "NoSuchAlloc" if a is None else self.alloc_problem(a)
                if problem:
                    with self.lock:
                        self.st.update(error=f"restart: {alloc}: {problem}")
                    return
                bits, name = a["bits"], ""
                take = dict(no_take(), alloc=alloc, bank_ids={w: a["banks"][w]["id"] for w in a["banks"]}, usage_from=alloc,
                            base=usage_norm(a.get("usage"), a.get("saved_unix", 0)))
            if not self.op_lock.acquire(timeout=60):
                with self.lock:
                    self.st.update(error="restart: a start is being admitted")
                return
            try:
                code, ans = self.begin_start_locked(bits, name, asked["kv"], asked["drafter"], asked["context"], asked["store"],
                                                    asked["reserve_gib"], take, asked["multi"], asked["multi_wait"])
            finally:
                self.op_lock.release()
            if code != 200:
                why = ans.get("error")
                why = why.get("message") if isinstance(why, dict) else why
                log(f"restart: the start was refused ({code}): {why}")
                with self.lock:
                    self.st.update(error=f"restart: {why}")
        except Exception as e:  # a bug here must not leave the state at stopping
            log(f"restart failed (launcher): {e!r}")
            with self.lock:
                self.st.update(state="failed" if self.st["state"] == "stopping" else self.st["state"], error=f"launcher: {e!r}")
        finally:
            with self.lock:
                self.st["restart"] = False

    def preflight(self, ts):
        nodes = self.probe_all()
        bad = [n["name"] for n in nodes if not n["reachable"]]
        if bad:
            raise Fail("not reachable over ssh: " + ", ".join(bad))
        bad = [f"{n['name']} ({n['gpu_error']})" for n in nodes if n["gpu_error"]]
        if bad:
            raise Fail("the GPU cannot be read on " + ", ".join(bad))
        busy = [f"{n['name']} (this lane's engine)" for n in nodes if n["engine"]]
        busy += [f"{n['name']} ({n['other']})" for n in nodes if n["other"]]
        if busy:
            raise Fail("the GPU is in use on " + ", ".join(busy))
        if self.cancel:
            raise Fail("cancelled")
        self.snapshot(ts + "-launch-pre-up")
        if self.cancel:
            raise Fail("cancelled")
        if not self.drop_caches:
            return
        # LAUNCH_DROP_CACHES=1: the page cache dropped on every node first (needs sudo without a password; the engine
        # drops its own weight files' pages either way)
        cmd = "sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"
        if self.dry:
            log(f"dry run: on every node: {cmd}")
        else:
            bad = [self.names[i] for i, (rc, _) in enumerate(self.on_all(cmd)) if rc != 0]
            if bad:
                raise Fail("the page cache could not be dropped (sudo -n) on " + ", ".join(bad))

    def bank_check(self, force=False):
        """Every width's bank (bank_list) on every node: its manifest and the layer files it lists, present at the
        manifest's size (stat only, the nodes in parallel, a few seconds). A width is available when every node has all
        of it. Kept BANK_TTL seconds (force: checked again now). {bits: {"id", "dir", "ok", "nodes": {name: "ok" | why}}}."""
        # the probe runs outside the lock, one at a time (a node whose stat hangs would hold the lock,
        # and /v1/launch/status with it, for the ssh's 60 s); a forced check runs even beside a background one
        force = force or self.banks_changed()  # banks.json edited since the last check: check now
        with self.bank_lock:
            if not force and self.bank_state and time.time() - self.bank_t < BANK_TTL:
                return self.bank_state
            if not force and self.bank_busy:
                return self.bank_state
            self.bank_busy = True
        try:
            return self.bank_probe()
        finally:
            with self.bank_lock:
                self.bank_busy = False

    def banks_changed(self):
        """Whether expert-plan/banks.json is another file than at the last check (its time or size)."""
        try:
            st = os.stat(self.banks)
            now = (st.st_mtime_ns, st.st_size)
        except OSError:
            now = None
        return now != getattr(self, "bank_stamp", None)

    def bank_probe(self):
        """One probe of every width's bank on every node (bank_check holds the single-flight flag)."""
        try:
            st = os.stat(self.banks)
            self.bank_stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            self.bank_stamp = None
        banks = self.bank_list()
        res = self.on_all(self.bank_cmd(banks), 60)
        state = {}
        for b in banks:
            nodes = {}
            for i, (rc, out) in enumerate(res):
                try:
                    nodes[self.names[i]] = json.loads(out.strip().splitlines()[-1]).get(str(b["bits"]), "not checked")
                except (ValueError, IndexError):
                    nodes[self.names[i]] = "not reachable" if rc == 255 else f"the check failed (rc {rc})"
            state[b["bits"]] = {"id": b["id"], "dir": b["dir"], "ok": all(v == "ok" for v in nodes.values()), "nodes": nodes}
        for w, s in state.items():
            if not s["ok"]:
                log(f"bank {s['id']} ({w} bit) not usable: " + "; ".join(f"{n}: {v}" for n, v in s["nodes"].items() if v != "ok"))
        drafter = {}
        if self.drafter_path:
            for i, (rc, out) in enumerate(res):
                try:
                    drafter[self.names[i]] = json.loads(out.strip().splitlines()[-1]).get("dflash2", "not checked")
                except (ValueError, IndexError):
                    drafter[self.names[i]] = "not reachable" if rc == 255 else f"the check failed (rc {rc})"
        with self.bank_lock:
            self.bank_state, self.bank_t, self.drafter_state = state, time.time(), drafter
        return state

    def bank_cmd(self, banks):
        """The command that checks these banks (and DFlash2's model) on a node: one JSON line, {bits: "ok" | why}."""
        arg = base64.b64encode(json.dumps({"banks": [{"bits": b["bits"], "dir": b["dir"], "manifest": manifest_of(b)} for b in banks],
                                           "drafter": self.drafter_path}).encode()).decode()
        code = base64.b64encode(BANK_PROBE.encode()).decode()
        return f"python3 -c \"$(echo {code} | base64 -d)\" {arg}"

    def bank_items(self, nodes):
        """A check's banks row of each node (those of a check need not be the lane's): the widths whose bank the node
        has whole; a width missing is a warning (the panel's plans then go without it). None for a node not read."""
        banks = self.bank_list()
        if not banks:
            return [None] * len(nodes)
        cmd = self.bank_cmd(banks)
        with ThreadPoolExecutor(len(nodes)) as ex:
            res = list(ex.map(lambda n: self.ssh(n, cmd, 60), nodes))
        out = []
        for rc, o in res:
            try:
                got = json.loads(o.strip().splitlines()[-1])
            except (ValueError, IndexError):
                out.append(None)
                continue
            have = [b for b in banks if got.get(str(b["bits"])) == "ok"]
            lack = [b for b in banks if got.get(str(b["bits"])) != "ok"]
            value = ", ".join(str(b["bits"]) for b in have) + " bit" if have else "none"
            # the other widths are optional: only the width every start needs makes this row more than a fact
            needed = any(b["bits"] == REQUIRED_BITS for b in have)
            if needed and lack:
                value += " (no " + ", ".join(str(b["bits"]) for b in lack) + " bit)"
            out.append({"key": "banks", "state": "ok" if needed else "fail", "value": value,
                        "fix": "" if needed else f"{REQUIRED_BITS} bit is always needed: " + "; ".join(f"{b['bits']} bit ({b['id']}) at {b['dir']}: {got.get(str(b['bits']), 'not checked')}" for b in lack if b["bits"] == REQUIRED_BITS)})
        return out

    def bank_view(self):
        """The launch screen's view of the banks (status, plan): the widths and whether each can be used; checked again
        in the background once BANK_TTL has passed, so a status never waits on ssh."""
        with self.bank_lock:
            state, t, busy = self.bank_state, self.bank_t, self.bank_busy
        if (time.time() - t >= BANK_TTL or self.banks_changed()) and not busy:
            threading.Thread(target=self.bank_check, daemon=True).start()
        return {"checked_unix": int(t), "checking": busy, "bits": {str(w): {"id": s["id"], "ok": s["ok"],
                "why": "; ".join(f"{n}: {v}" for n, v in s["nodes"].items() if v != "ok")} for w, s in sorted(state.items())},
                "available": [w for w, s in sorted(state.items()) if s["ok"]], "required": REQUIRED_BITS}

    def missing_bits(self, bits, force=False):
        """The widths `bits` uses whose bank some node lacks (or that have no bank), with why."""
        state = self.bank_check(force)
        out = {}
        for w in sorted(set(bits)):
            s = state.get(w)
            if s is None:
                out[w] = "no bank in expert-plan/banks.json"
            elif not s["ok"]:
                out[w] = f"bank {s['id']}: " + "; ".join(f"{n}: {v}" for n, v in s["nodes"].items() if v != "ok")
        return out

    def bank_by_id(self):
        """Every bank of expert-plan/banks.json by its id (~ in dir and manifest expanded)."""
        out = {}
        for b in json.load(open(self.banks))["banks"]:
            for k in ("dir", "manifest"):
                if b.get(k):
                    b[k] = os.path.expanduser(b[k])
            out.setdefault(b["id"], b)
        return out

    def bank_dirs(self, ids=None):
        """TF_GLM53_EXPERT_BANKS: a directory a width, the lane's first bank of it or the bank `ids` names for it."""
        by_id = self.bank_by_id()
        ids = ids or {}
        return ",".join(f"{b['bits']}:{by_id[ids[b['bits']]]['dir'] if b['bits'] in ids else b['dir']}" for b in self.bank_list())

    def bank_list(self):
        """One bank a width from expert-plan/banks.json (the first listed with a directory: a blank one is that width
        left unused, kept in banks.json so the launch settings can name one again)."""
        seen, out = set(), []
        for b in json.load(open(self.banks))["banks"]:
            if not b.get("dir"):
                continue
            if b["bits"] not in seen:
                seen.add(b["bits"])
                for k in ("dir", "manifest"):  # ~: the home directory, the same on every node
                    if b.get(k):
                        b[k] = os.path.expanduser(b[k])
                out.append(b)
        return out

    def write_plan(self, bits, name, ts, ids=None, note="tf_launcher"):
        """The bits as an expert plan (expert_plan.py mix): each width from the lane's first bank of it, or the bank
        `ids` names for it (a saved allocation's), copied to every node at the same path."""
        banks = {b["bits"]: b["id"] for b in self.bank_list()}
        banks.update(ids or {})
        used = sorted(set(bits))
        miss = [b for b in used if b not in banks]
        if miss:
            raise Fail("no bank for " + ", ".join(f"{b} bit" for b in miss))
        raw = os.path.join(self.dir, f"plan-{ts}.bits")
        out = os.path.join(self.dir, f"plan-{ts}.json")
        open(raw, "wb").write(bits)
        tool = os.path.join(self.pkg_local, "tools", "quant", "expert_plan.py")
        cmd = [sys.executable, tool, "--banks", self.banks, "mix", "--bits", raw, "--map", ",".join(f"{b}:{banks[b]}" for b in used),
               "--out", out, "--name", name or f"launch {ts}", "--note", note]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise Fail("the plan could not be written: " + (r.stderr or r.stdout).strip()[-300:])
        log(r.stdout.strip())
        # every rank reads the plan: the same file at the same path on every node, checked by its sha256
        there = os.path.join(self.pkg, "launch", os.path.basename(out))
        want = hashlib.sha256(open(out, "rb").read()).hexdigest()
        if self.dry:
            log(f"dry run: {out} to {there} on every node")
            return there
        def put(node):
            try:  # a hung node fails the copy instead of the start waiting for ever
                if subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, f"mkdir -p {os.path.dirname(there)}"], capture_output=True, timeout=60).returncode != 0:
                    return False
                if subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", out, f"{node}:{there}"], capture_output=True, timeout=120).returncode != 0:
                    return False
            except subprocess.TimeoutExpired:
                return False
            return self.ssh(node, f"sha256sum {there}")[1][:64] == want
        with ThreadPoolExecutor(self.world) as ex:
            ok = list(ex.map(put, self.nodes))
        bad = [self.names[i] for i, x in enumerate(ok) if not x]
        if bad:
            raise Fail("the plan could not be copied to " + ", ".join(bad))
        return there

    def rank_errors(self):
        """The first `error: glm53:` line of each rank's log after a start failed (the reason the engine gave; the log's
        last lines are often the debug allocator's leak reports at exit), the distinct ones."""
        ranks = " ".join(f"rank{r}" for r in range(len(self.nodes)))
        seen, out = set(), []
        for i, (rc, line) in enumerate(self.on_all(f"for s in {ranks}; do {self.keep} cat $s 2>/dev/null; done | grep -m1 -h '^error: glm53'", 20)):
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                out.append(f"{self.names[i]}: {line}")
        return " / ".join(out[:3])

    def run_tf(self, what, env, ts, extra=None, drafter=None):
        """tf-glm53 engine-up / engine-down (what: up or down; up with -d DRAFTER), the engine's own start and stop
        (tf-glm53 up starts this launcher, and its start and down come here), its output into the launcher's log (in
        memory, the lane's keeper: tf-glm53 logs launcher); the exit code and the last lines."""
        cmd = ["bash", os.path.join(self.pkg_local, "tf-glm53"), "-c", self.conf] + (["-d", drafter] if drafter else []) + ["engine-" + what]
        if self.dry:
            log(f"dry run: LAUNCH_ENV={shlex.quote(env)} {' '.join(f'{k}={v}' for k, v in (extra or {}).items())} {' '.join(cmd)}")
            raise Fail("dry run: nothing was started" if what == "up" else "dry run: nothing was stopped")
        lines = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=dict(os.environ, LAUNCH_ENV=env, **(extra or {})))
        with self.lock:
            self.proc = proc

        def read():
            for line in proc.stdout:
                sys.stdout.write(f"[{what} {ts}] {line}")
                sys.stdout.flush()
                lines.append(line.rstrip())
                if what == "up" and line.startswith("started; waiting"):
                    self.set_phase("weights")
                if what == "up" and line.startswith("ready after"):
                    self.set_phase("ready")
        t = threading.Thread(target=read, daemon=True)
        t.start()
        while proc.poll() is None:
            if what == "up" and self.st["phase"] not in ("checking", "ready"):
                self.follow_phase()
            time.sleep(0.25)
        t.join()
        rc = proc.returncode
        with self.lock:
            self.proc = None
        keep = [x.strip() for x in lines if x and not re.fullmatch(r"\s*\d+s", x)]
        # the reason, not the notes every start prints (the open API, where the logs are, the replies' limits, the
        # drafter): a check's NG lines first, else the last other lines (11-01: a start refused for the workers' files
        # read "no key" in the panel)
        notes = re.compile(r"(note|logs|replies|drafter|ring):|\(DRAFTER ")
        ng = [x for x in keep if re.search(r":\s*NG\b", x)]
        rest = [x for x in keep if not notes.match(x)]
        return rc, " / ".join(ng[:4] + [x for x in rest[-2:] if x not in ng] if ng else rest[-4:])

    def follow_phase(self):
        p = self.read_progress()
        if p and p.get("phase") in PHASES:
            self.set_phase(p["phase"])

    def measured(self, bits, before, drafter=None):
        """After a start: each node's memory the engine took and rank 0's start-up steps, for later plans and ETAs."""
        time.sleep(5)
        after = {n["name"]: n["mem_available"] for n in self.probe_all()}
        try:
            mem = json.loads(self.engine_get("/v1/memory", 5)[1])
        except (OSError, ValueError):
            return
        steps = {s["label"]: s["bytes"] for s in mem.get("startup", [])}
        ex = experts_bytes(bits, self.world)
        other = steps.get("layers (weights, routed experts included)", 0) - mem.get("experts", 0) + \
            steps.get("the DCP partner's q_b / wk (PEER_Q)", 0) + steps.get("embed + lm_head + final norm", 0) + steps.get("MTP layer", 0)
        kv = steps.get("KV caches (the window)", 0)
        # the conversations decoded together: the work this start shows holds their idle memory, which a plan adds back
        # for its own N (plan, multi_bytes); kept less it, the history reads as MULTI=0 whatever N each start had
        with self.lock:
            multi = dict(self.st.get("multi") or {"n": self.multi_conf, "wait": self.multi_wait_conf})
        m_idle = self.multi_bytes(multi["n"], drafter or self.drafter_conf)[0]
        nodes = {}
        for name, a0 in before.items():
            took = a0 - after.get(name, a0)
            nodes[name] = {"avail_before": a0, "avail_after": after.get(name, 0), "other_weights": other, "kv": kv,
                           "work": max(0, took - ex - other - kv - m_idle)}
        with self.lock:
            seen = dict(self.st["seen"])
        seen["done"] = time.time()
        phases = {k: seen[PHASES[i + 1] if i + 1 < len(PHASES) else "done"] - seen[k]
                  for i, k in enumerate(PHASES) if k in seen and (PHASES[i + 1] if i + 1 < len(PHASES) else "done") in seen}
        with self.lock:
            kvu = self.st.get("kv")
        doc = {"unix": int(time.time()), "world": self.world, "nodes": nodes, "phases": phases, "kv": kvu,
               "drafter": drafter or self.drafter_conf, "multi": multi}
        self.record_start(doc)
        self.keep_usage()

    def record_start(self, doc):
        """last-start.json and last-start-<drafter>.json (the plans of later starts with this drafter), with the
        drafter's measured starts as history: each one's positions (context + store) and each node's work, the latest
        at a position count kept (START_HISTORY at most; another node count's dropped)."""
        old = self.read_json(f"last-start-{doc['drafter']}.json") or {}
        hist = list(old.get("history") or ([start_entry(old)] if old.get("nodes") else [])) if old.get("world") == doc["world"] else []
        new = start_entry(doc)
        hist = [h for h in hist if h and h.get("unix") != new["unix"] and h.get("positions") != new["positions"]] + [new]
        doc["history"] = hist[-START_HISTORY:]
        self.write_json("last-start.json", doc)
        self.write_json(f"last-start-{doc['drafter']}.json", doc)

    def begin_stop(self, body=b""):
        try:
            save = json.loads(body or b"{}").get("save_usage")
        except (ValueError, AttributeError):
            raise Fail("bad JSON")
        if save is not None and not isinstance(save, bool):  # 1 and 0 are not true and false
            raise Fail("save_usage: true, false or absent")
        with self.lock:
            s = self.st["state"]
            if s == "stopped":
                return 409, {"error": "already stopped"}
            if s == "starting" and self.job and self.job.is_alive():
                self.cancel = True
                self.st["state"] = "stopping"
                if self.proc:
                    self.proc.send_signal(signal.SIGTERM)  # tf-glm53 engine-up stops every rank on TERM
                return 200, {"state": "stopping"}
            if self.job and self.job.is_alive():
                return 409, {"error": f"busy ({s})"}
            self.st.update(state="stopping", phase="", t0=time.time())
            self.job = threading.Thread(target=self.run_stop, args=(save,), daemon=True)
            self.job.start()
        return 200, {"state": "stopping"}

    def run_stop(self, save=None):
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            if self.engine_up:
                self.keep_usage()
            self.snapshot(ts + "-launch-pre-down")
            rc, tail = self.run_tf("down", "", ts)
            if rc == 0:
                self.end_run(None, save)  # save_usage: true / false / absent = pending
                self.wait_memory_back()
                # the engine settings saved while it ran, read before the state says stopped (a plan asked at once
                # takes them)
                if self.conf_mtimes() != self.conf_seen:
                    self.reload_conf()
            with self.lock:
                self.st.update(state="stopped" if rc == 0 else "failed", phase="", bits=None, set="",
                               **(no_take() if rc == 0 else {}),
                               error="" if rc == 0 else tail)
        except Fail as e:
            with self.lock:
                self.st.update(state="failed", phase="", error=str(e))
        except Exception as e:
            log(f"stop failed (launcher): {e!r}")
            with self.lock:
                self.st.update(state="failed", phase="", error=f"launcher: {e!r}")
        finally:
            self.snapshot(ts + "-launch-post-down")

    def wait_memory_back(self, limit=30):
        """After a stop: every node's memory back (MemAvailable within 3 GiB of before the last start, or two readings
        alike when there is none), at most `limit` s, so a plan made right after the stop reads freed memory."""
        with self.lock:
            self.engine_up = False
            ref = dict(self.start_before)
        if not ref:
            ref = {k: v["avail_before"] for k, v in ((self.read_json("last-start.json") or {}).get("nodes") or {}).items()}
        t0, prev = time.time(), None
        while time.time() - t0 < limit:
            now = {n["name"]: n["mem_available"] for n in self.probe_all()}
            if ref and all(now.get(k, 0) >= v - 3 * GiB for k, v in ref.items()):
                return
            if not ref and prev and all(abs(now[k] - prev.get(k, 0)) < GiB // 2 for k in now):
                return
            prev = now
            time.sleep(1)
        log(f"memory after the stop not back within {limit} s: " + ", ".join(f"{k} {v / GiB:.1f} GiB" for k, v in now.items()))

    # ---- saved allocations: ATLAS_DIR/allocs/user/<name>.json (glm53-expert-alloc v1) ----

    def alloc_path(self, name):
        return os.path.join(self.atlas, "allocs", "user", name + ".json")

    def alloc_names(self):
        """The names of the files in allocs/user (those that do not parse too)."""
        try:
            return [f[:-5] for f in os.listdir(os.path.join(self.atlas, "allocs", "user")) if f.endswith(".json")]
        except FileNotFoundError:
            return []

    def alloc_read(self, name):
        """A saved allocation checked: glm53-expert-alloc v1 of this model's shape, bits 2 to 5, a bank (id, manifest
        sha256) for every width it uses; bits as bytes and banks by width. None when there is none or it does not hold
        (logged)."""
        if not ALLOC_NAME.fullmatch(name or "") or name not in self.alloc_names():  # the name exactly, on any file system
            return None
        try:
            a = json.load(open(self.alloc_path(name)))
            if a.get("format") != "glm53-expert-alloc" or a.get("version") != 1 or (a.get("first_layer"), a.get("layers"), a.get("experts")) != (FIRST, L, E):
                raise ValueError(f"not glm53-expert-alloc v1 of {L} x {E} from layer {FIRST}")
            b = base64.b64decode(a["bits"], validate=True)
            if len(b) != L * E or any(v < 2 or v > 5 for v in b):
                raise ValueError("bits: length or value")
            banks = {int(w): v for w, v in a["banks"].items()}
            if set(b) - set(banks) or any(not isinstance(banks[w].get("id"), str) or not isinstance(banks[w].get("manifest_sha256"), str) for w in set(b)):
                raise ValueError("a width without its bank")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            log(f"allocation {name}: left out ({e})")
            return None
        a["bits"], a["banks"] = b, {w: banks[w] for w in set(b)}
        return a

    def alloc_problem(self, a, lane=None):
        """Why an allocation cannot start on this lane: a bank of a width it uses that banks.json lacks (or has for
        another width): "BankMissing <id>"; one banks.json has with another manifest: "BankDiffers <id>"; "" when it can."""
        lane = lane if lane is not None else self.bank_by_id()
        for w in sorted(a["banks"]):
            mine = a["banks"][w]
            have = lane.get(mine["id"])
            if not have or have.get("bits") != w:
                return "BankMissing " + mine["id"]
            if have.get("manifest_sha256") != mine["manifest_sha256"]:
                return "BankDiffers " + mine["id"]
        return ""

    def alloc_view(self, name, a, lane, full=False):
        """GET /v1/launch/allocs' item (full: /v1/launch/alloc's, with the bits, the shape, each bank's manifest and this
        lane's directory of it, and the usage with its totals). counts, mean_bits, valid and problem computed here."""
        b, used = a["bits"], sorted(a["banks"])
        problem = self.alloc_problem(a, lane)
        u = usage_norm(a.get("usage"), a.get("saved_unix", 0))
        out = {"name": name, "saved_unix": int(a.get("saved_unix") or 0), "source": str(a.get("source", "")), "basis": str(a.get("basis", "")),
               "note": str(a.get("note", "")), "counts": {str(w): b.count(w) for w in used}, "mean_bits": round(sum(b) / len(b), 4),
               "banks": [{"bits": w, "id": a["banks"][w]["id"]} for w in used], "valid": not problem, "problem": problem,
               "usage": {k: v for k, v in u.items() if k != "total"} if u else None}  # list: without the totals
        if full:
            def lane_dir(w):
                have = lane.get(a["banks"][w]["id"])
                return have["dir"] if have and have.get("bits") == w else ""
            out.update(first_layer=FIRST, layers=L, experts=E, bits=base64.b64encode(b).decode(), usage=u,
                       banks=[{"bits": w, "id": a["banks"][w]["id"], "manifest_sha256": a["banks"][w]["manifest_sha256"], "dir": lane_dir(w)} for w in used])
        return out

    def allocs_list(self):
        """GET /v1/launch/allocs: the saved allocations, newest first, without their bits (a file that does not hold is
        left out)."""
        lane = self.bank_by_id()
        items = [self.alloc_view(n, a, lane) for n in self.alloc_names() for a in [self.alloc_read(n)] if a is not None]
        items.sort(key=lambda x: (-x["saved_unix"], x["name"]))
        return {"world": self.world, "allocs": items}

    def alloc_one(self, name):
        """GET /v1/launch/alloc?name=NAME: one saved allocation with its bits (the name exactly)."""
        a = self.alloc_read(name)
        if a is None:
            return 404, {"error": "NoSuchAlloc"}
        return 200, self.alloc_view(name, a, self.bank_by_id(), full=True)

    def alloc_in_use(self, name):
        """The engine was started with this allocation and is not stopped (nor failed)."""
        with self.lock:
            return self.st.get("alloc") == name and self.st["state"] in ("starting", "running", "stopping")

    def alloc_write(self, path, obj, replace=False):
        """`obj` written beside `path` and linked into place (never over another file: FileExistsError), or moved over
        it (replace)."""
        tmp = path + f".tmp{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"  # unique
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        try:
            if replace:
                os.replace(tmp, path)
            else:
                os.link(tmp, path)  # fails when the name exists: never over a saved one
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def alloc_save(self, name, bits, source="", note="", basis=None, engine=False, with_usage=False, replace=False):
        """POST /v1/launch/allocs/save: an allocation as glm53-expert-alloc v1 in ATLAS_DIR/allocs/user/<name>.json:
        the bits, a bank (id, manifest sha256) a width used (the lane's first of each width), the shape; never over a
        name, in any case (409 Exists), unless replace: the old version (that name in any case) moved to allocs/trash/;
        when its bits are the bits saved, the new file keeps its source and basis, and without a note its note (kept).
        engine: the running engine's bits (those sent are not used); with_usage: the usage kept with them = the usage
        the start took over (its allocation's) + this run's picks since its start (the launcher's snapshot,
        last-usage.json), so two saves in one run count the run once; both 409 NotRunning while the engine does not
        run."""
        if not ALLOC_NAME.fullmatch(name or ""):
            raise Fail("BadRequest: name: letters, digits, _ . - (64 at most)")
        if len(str(note or "")) > NOTE_MAX:
            raise Fail(f"BadRequest: note: {NOTE_MAX} characters at most")
        if basis is not None and basis not in BASES:
            raise Fail("BadRequest: basis: decode, prompt or both")
        usage = None
        if engine or with_usage:
            with self.lock:
                running = self.st["state"] == "running" and self.engine_up
            if not running:
                return 409, {"error": "NotRunning"}
            self.keep_usage()  # this run's picks and bits, read now
            snap = self.read_json("last-usage.json")
            if not snap or run_state(snap) != "running" or snap.get("run") != self.run_id():
                return 502, {"error": "the running engine's usage could not be read"}
            if engine:
                if not snap.get("bits"):
                    return 502, {"error": "the running engine's bits could not be read"}
                bits = base64.b64decode(snap["bits"])
            if with_usage:
                with self.lock:
                    base = self.st.get("base")
                usage = {"rows": {"decode": snap["rows"]["decode"] + (base["rows"]["decode"] if base else 0),
                                  "prompt": snap["rows"].get("prompt", 0) + (base["rows"]["prompt"] if base else 0)},
                         "total": {"decode": add_u64(base["total"]["decode"] if base else "", snap["total"]["decode"])},
                         "runs": (base["runs"] if base else 0) + 1,
                         "since_unix": (base["since_unix"] if base and base["since_unix"] else 0) or int(snap.get("started_unix", 0)),
                         "saved_unix": int(time.time())}
                if snap["total"].get("prompt") or (base and base["total"].get("prompt")):
                    usage["total"]["prompt"] = add_u64(base["total"].get("prompt", "") if base else "", snap["total"].get("prompt", ""))
        if bits is None or len(bits) != L * E or any(v < 2 or v > 5 for v in bits):
            raise Fail("BadRequest: bits: length or value")
        banks = {b["bits"]: b for b in self.bank_list()}
        if engine:  # the banks the running start was given (a saved allocation's), the lane's first for the others
            with self.lock:
                ids = dict(self.st.get("bank_ids") or {}) if self.st.get("alloc") else {}
            by_id = self.bank_by_id()
            banks.update({w: by_id[i] for w, i in ids.items() if i in by_id})
        miss = sorted({v for v in bits} - set(banks))
        if miss:
            raise Fail("no bank for " + ", ".join(f"{b} bit" for b in miss))
        obj = {"format": "glm53-expert-alloc", "version": 1, "name": name, "saved_unix": int(time.time()),
               "first_layer": FIRST, "layers": L, "experts": E,
               "banks": {str(w): {"id": banks[w]["id"], "manifest_sha256": banks[w]["manifest_sha256"]} for w in sorted(set(bits))},
               "bits": base64.b64encode(bytes(bits)).decode()}
        if source:
            obj["source"] = str(source)[:32]
        if note:
            obj["note"] = str(note)
        if basis is not None:
            obj["basis"] = basis
        if usage:
            obj["usage"] = usage
        os.makedirs(os.path.dirname(self.alloc_path(name)), exist_ok=True)
        with self.alloc_lock:
            old = next((n for n in self.alloc_names() if n.lower() == name.lower()), None)
            if old is not None and not replace:
                return 409, {"error": "Exists"}
            # over an allocation of the same bits: how they were made stays (source, basis); no note keeps its note
            kept = []
            prev = self.alloc_read(old) if old is not None else None
            if prev is not None:
                if prev["bits"] == bytes(bits):
                    for k in ("source", "basis"):
                        obj.pop(k, None)
                        if prev.get(k):
                            obj[k] = prev[k]
                        kept.append(k)
                if not note and prev.get("note"):
                    obj["note"] = prev["note"]
                    kept.append("note")
            path = self.alloc_path(name)
            tmp = path + f".tmp{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"  # unique
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=1)
            try:
                trash = self.alloc_trash(old) if old is not None else ""  # written first: the old one moves only then
                os.link(tmp, path)
            except FileExistsError:
                return 409, {"error": "Exists"}
            finally:
                os.unlink(tmp)
        log(f"allocation {name} saved ({source or '-'}{', engine' if engine else ''}"
            f"{', with usage of ' + str(usage['runs']) + ' runs' if usage else ''}{', replacing ' + old + ' (' + trash + ')' if old else ''}"
            f"{', kept ' + '/'.join(kept) if kept else ''}): "
            f"{dict(sorted((str(w), bits.count(w)) for w in set(bits)))}")
        return 200, dict({"saved": name, "replaced": old is not None}, **({"kept": kept} if kept else {}))

    def alloc_trash(self, name):
        """The file of allocation `name` moved to ATLAS_DIR/allocs/trash/<name>-<unix>.json (-<unix>.1 and on within one
        second); its path under ATLAS_DIR. Called under alloc_lock."""
        trash = os.path.join(self.atlas, "allocs", "trash")
        os.makedirs(trash, exist_ok=True)
        t, k = int(time.time()), 0
        while os.path.exists(os.path.join(trash, f"{name}-{t}{'.' + str(k) if k else ''}.json")):
            k += 1
        dest = f"{name}-{t}{'.' + str(k) if k else ''}.json"
        os.rename(self.alloc_path(name), os.path.join(trash, dest))
        return "allocs/trash/" + dest

    def alloc_rename(self, raw):
        """POST /v1/launch/allocs/rename {name, to}: the file and the name in it; a name differing only in case from its
        own is allowed, another allocation's (in any case) is not (409 Exists); not the running start's (409 InUse)."""
        try:
            j = json.loads(raw or b"{}")
            name, to = j["name"], j["to"]
        except (ValueError, KeyError, TypeError):
            return 400, {"error": "BadRequest"}
        if not isinstance(name, str) or not isinstance(to, str) or not ALLOC_NAME.fullmatch(to):
            return 400, {"error": "BadRequest"}
        with self.alloc_lock:
            if not ALLOC_NAME.fullmatch(name) or name not in self.alloc_names():
                return 404, {"error": "NoSuchAlloc"}
            if to == name:
                return 200, {"renamed": to}
            if any(n.lower() == to.lower() and n != name for n in self.alloc_names()):
                return 409, {"error": "Exists"}
            if self.alloc_in_use(name):
                return 409, {"error": "InUse"}
            src, dst = self.alloc_path(name), self.alloc_path(to)
            try:
                obj = json.load(open(src))
                obj["name"] = to
            except (OSError, ValueError, TypeError) as e:
                return 500, {"error": f"Failed: {e}"}
            if to.lower() == name.lower():  # the case only: in place, then the file renamed (one file on any file system)
                self.alloc_write(src, obj, replace=True)
                os.rename(src, dst)
            else:
                try:
                    self.alloc_write(dst, obj)
                except FileExistsError:
                    return 409, {"error": "Exists"}
                os.unlink(src)
        log(f"allocation {name} renamed {to}")
        return 200, {"renamed": to}

    def alloc_delete(self, raw):
        """POST /v1/launch/allocs/delete {name}: the file moved to ATLAS_DIR/allocs/trash/<name>-<unix>.json (put back by
        hand), not removed; not the running start's (409 InUse)."""
        try:
            name = json.loads(raw or b"{}")["name"]
        except (ValueError, KeyError, TypeError):
            return 400, {"error": "BadRequest"}
        if not isinstance(name, str):
            return 400, {"error": "BadRequest"}
        with self.alloc_lock:
            if not ALLOC_NAME.fullmatch(name) or name not in self.alloc_names():
                return 404, {"error": "NoSuchAlloc"}
            if self.alloc_in_use(name):
                return 409, {"error": "InUse"}
            dest = self.alloc_trash(name)
        log(f"allocation {name} deleted: {dest}")
        return 200, {"deleted": name, "trash": dest}

    def resort(self, body):
        """POST /v1/launch/resort: the experts re-sorted by this run's picks (the saved runs too only with
        include_saved true) with the counts by bits kept: each expert's share of its layer's
        picks, every expert ranked by it (as the launch screen ranks them), the widest bits to the first.
        While the engine runs: its own picks and bits; apply asks the engine for the swap and saves the allocation as
        sort-<time> (save false: nothing saved, the panel names one itself). While it does not: the last run's picks and
        bits (launch/last-usage.json, whatever became of it: saved, pending or discarded, said in basis.run); apply saves
        the allocation (applied false, saved, for the next start). An engine that cannot swap (TF_GLM53_MULTI): the
        allocation saved and 409 asking for a stop and a start with it."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        include = bool(j.get("include_current", True))
        with_saved = bool(j.get("include_saved", False))
        # save false: nothing saved (the panel names an allocation itself, POST /v1/launch/allocs/save)
        keep = j.get("save", True)
        if not isinstance(keep, bool):
            raise Fail("save: true or false")
        # include_alloc: the usage the start took over (the allocation it was started from or took the usage of)
        # added to this run's picks, while the engine runs
        with_alloc = j.get("include_alloc", False)
        if not isinstance(with_alloc, bool):
            raise Fail("include_alloc: true or false")
        run = None
        if self.engine_up:
            s, raw = self.engine_get("/v1/experts/usage?window=10", 10)
            if s != 200:
                return 502, {"error": f"the engine's usage: {s}"}
            u = json.loads(raw)
        else:
            u = self.read_json("last-usage.json")
            if not u or not u.get("bits") or not isinstance(u.get("total"), dict):
                return 412, {"error": "the engine is not running and no run was recorded to sort by"}
            run = {"run": u.get("run"), "state": run_state(u), "started_unix": u.get("started_unix", 0), "ended_unix": u.get("saved_unix", 0)}
            if with_saved and run["state"] == "saved":
                include = False
        cur = bytearray(base64.b64decode(u["bits"]))
        base, ufrom = None, ""
        if with_alloc and run is None:
            with self.lock:
                base, ufrom = self.st.get("base"), self.st.get("usage_from", "")
        basis = j.get("basis", "both")  # both: decode weighed DECODE_WEIGHT, prefill the rest
        if basis not in BASES:
            raise Fail("basis: decode, prompt or both")
        # each source's share of its layer (an expert's picks / the layer's picks); "both" weighs the sources that have
        # rows, decode by DECODE_WEIGHT and prompt by the rest (one source alone counts in full)
        share = [0.0] * (L * E)
        weight = 0.0
        got = {}
        for src in ("decode", "prompt") if basis == "both" else (basis,):
            counts, rows, runs = self.learning(u if include else None, src, with_saved)
            if base and base["total"].get(src):
                counts = [x + y for x, y in zip(counts, struct.unpack(f"<{L * E}Q", base64.b64decode(base["total"][src])))]
                rows += base["rows"].get(src, 0)
            got[src] = (rows, runs)
            if rows == 0:
                continue
            w = (DECODE_WEIGHT if src == "decode" else 1 - DECODE_WEIGHT) if basis == "both" else 1.0
            weight += w
            for li in range(L):
                tot = sum(counts[li * E:(li + 1) * E])
                if tot:
                    for e in range(E):
                        share[li * E + e] += w * counts[li * E + e] / tot
        if weight == 0:
            return 412, {"error": f"no usage to sort by ({basis}{', the saved runs' if with_saved else ', this run'})"}
        share = [x / weight for x in share]
        rows, runs = got.get("decode", (0, 0)) if basis != "prompt" else (0, 0)
        prows, pruns = got.get("prompt", (0, 0))
        frozen = set()
        if not self.generic:  # the all-4-bit kernel's layers take 4 bits only
            frozen = {li for li in range(L) if all(b == 4 for b in cur[li * E:(li + 1) * E])}
        movable = [i for i in range(L * E) if i // E not in frozen]
        # ties keep the bits each expert has, so an unused expert does not move for nothing
        movable.sort(key=lambda i: (-share[i], -cur[i], i))
        want = sorted((cur[i] for i in movable), reverse=True)
        target = bytearray(cur)
        for i, b in zip(movable, want):
            target[i] = b
        up = sum(1 for a, b in zip(cur, target) if b > a)
        down = sum(1 for a, b in zip(cur, target) if b < a)
        layers = len({i // E for i in range(L * E) if cur[i] != target[i]})
        out = {"target_bits": base64.b64encode(bytes(target)).decode(), "changed": up + down, "up": up, "down": down,
               "counts": {str(w): target.count(w) for w in sorted(set(target))}, "frozen_layers": len(frozen),
               "layers": layers, "basis": {"source": basis, "rows": rows, "runs": runs if basis != "prompt" else pruns, "prompt_rows": prows,
                                         "prompt_runs": pruns, "include_current": include, "include_saved": with_saved,
                                         "current_rows": u["rows"]["decode"] if include and basis != "prompt" else 0,
                                         "current_prompt_rows": u["rows"].get("prompt", 0) if include and basis != "decode" else 0,
                                         "engine": "running" if run is None else "stopped", "run": run,
                                         "alloc": {"name": ufrom, "rows": base["rows"]["decode"], "prompt_rows": base["rows"]["prompt"],
                                                   "runs": base["runs"]} if base else None},
               # measured (TP4): 64 experts over 2 layers in 983 ms, 118 over 63 in 8.2 s, 9,756 over 75 in 83-90 s
               "est_ms": int(layers * 115 + (up + down) * 8) if up + down else 0}
        # the widths it moves experts to (stopped: every width, the next start loads them all): a bank some node lacks
        # refuses the re-sort before the engine is asked
        miss = self.missing_bits(set(target) if run is not None else {b for a, b in zip(cur, target) if a != b})
        out["missing_bits"] = {str(w): why for w, why in miss.items()}
        # while the engine runs and can swap: whether every node has the memory the swap needs (the engine refuses it
        # otherwise); an engine that swaps nothing while it runs (TF_GLM53_MULTI: "available" false) only saves
        live = run is None and up + down > 0
        if live:
            s, raw = self.engine_get("/v1/experts/swap", 5)
            live = s == 200 and bool(json.loads(raw).get("available", True))
        mem = self.swap_memory(cur, target) if live else None
        if mem is not None:
            out["memory"] = mem
        if not j.get("apply"):
            return 200, out
        if miss:
            return 412, dict(out, error="no data for " + ", ".join(f"{w} bit ({why})" for w, why in miss.items()))
        if mem is not None and not mem["fits"]:
            return 412, dict(out, error=mem["why"])
        what = f"re-sort by {basis} ({'saved runs + ' if with_saved else ''}{'the usage of ' + ufrom + ' + ' if base else ''}{'this run' if run is None else 'the last run ' + str(run['run']) + ' (' + run['state'] + ')'}): {rows} decode rows, {prows} prompt rows"
        if run is not None:
            saved = ""
            if keep:
                code, res = self.alloc_save(time.strftime("sort-%Y%m%d-%H%M%S"), bytes(target), "sort", what[:NOTE_MAX], basis)
                if code != 200:
                    return code, dict(out, error=res.get("error", "the allocation was not saved"))
                saved = res.get("saved", "")
            log(f"re-sort while stopped ({basis}, the last run {run['run']}, {run['state']}): {up + down} experts ({up} up, {down} down) in {layers} layers; saved {saved or '-'}")
            return 200, dict(out, applied=False, saved=saved)
        if up + down == 0:
            return 200, dict(out, applied=False)
        c = http.client.HTTPConnection(*self.engine, timeout=30)
        try:
            c.request("POST", "/v1/experts/swap", body=json.dumps({"bits": out["target_bits"]}), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            st, sw = r.status, r.read()
        finally:
            c.close()
        if st == 409 and b"UnsupportedWithMulti" in sw:
            # TF_GLM53_MULTI swaps nothing while it runs: the allocation kept for a start after a stop (save false:
            # not kept, the panel saves the estimate under a name of its own)
            saved = ""
            if keep:
                code, res = self.alloc_save(time.strftime("sort-%Y%m%d-%H%M%S"), bytes(target), "sort", what[:NOTE_MAX], basis)
                saved = res.get("saved", "") if code == 200 else ""
            return 409, dict(out, applied=False, saved=saved, multi=True,
                             error="this engine runs TF_GLM53_MULTI and changes no bits while it runs" +
                                   (": stop it and start with the saved allocation" if keep else ""))
        if st == 409:
            return 409, {"error": "busy"}
        if st != 200:
            return st, {"error": f"the engine's swap: {sw.decode(errors='replace')[:200]}"}
        saved = ""
        if keep:
            try:
                code, res = self.alloc_save(time.strftime("sort-%Y%m%d-%H%M%S"), bytes(target), "sort", what[:NOTE_MAX], basis)
                saved = res.get("saved", "")
            except (Fail, OSError) as e:
                log(f"re-sort allocation not saved: {e}")
        log(f"re-sort ({basis}, {'saved runs + this run' if with_saved and include else 'saved runs' if with_saved else 'this run'}): {up + down} experts ({up} up, {down} down) in {layers} layers by {rows} decode rows, {prows} prompt rows; saved {saved or '-'}")
        return 200, dict(out, applied=True, saved=saved, swap=json.loads(sw))

    def swap_need(self, cur, target):
        """What a live swap from `cur` to `target` needs available on every rank, as the engine's checkMemory counts it
        (expert_swap.zig): the changed layers in the engine's order (those whose bits do not grow first, then those that
        grow), each layer's new slabs and reads on top of what the layers before it grew, the largest of those, and the
        margin. With that layer, its slabs and the growth before it. None on three nodes (unequal shares a rank)."""
        if S % self.world:
            return None
        unit = 3 * D * (I // self.world) // 8  # bytes a bit, an expert, a rank (expertUnit)
        grow = peak = 0
        at = (0, 0, 0)
        for grows in (False, True):
            for li in range(L):
                c, t = cur[li * E:(li + 1) * E], target[li * E:(li + 1) * E]
                d = sum(t) - sum(c)
                if c == t or (d > 0) != grows:
                    continue
                slabs = sum(unit * b * (2 if b != a else 1) for a, b in zip(c, t))
                g = max(grow, 0)
                if g + slabs > peak:
                    peak, at = g + slabs, (FIRST + li, slabs, g)
                grow += unit * d
        return peak + self.swap_margin, at

    def swap_memory(self, cur, target):
        """Whether a live swap fits on every node now: the estimate's "memory". Each node's MemAvailable as the node
        poll read it in the last 10 s, else read again; a node not read (unreachable) counts as unknown, not short
        (the engine checks it itself)."""
        got = self.swap_need(cur, target)
        if got is None:
            return None
        need, (layer, slabs, grown) = got
        with self.lock:
            nodes, t = list(self.node_info), self.node_t
        if not nodes or time.time() - t > 10:  # read for this estimate only (the node poll keeps node_info)
            with ThreadPoolExecutor(self.world) as ex:
                nodes = list(ex.map(self.probe, range(self.world)))
        read = lambda n: n.get("reachable") and n.get("mem_available")
        ranks = [{"node": n.get("name", ""), "available_mib": n["mem_available"] >> 20 if read(n) else None,
                  "fits": n["mem_available"] >= need if read(n) else None} for n in nodes]
        short = [n for n in nodes if read(n) and n["mem_available"] < need]
        why = ""
        if short:
            n = min(short, key=lambda n: n["mem_available"])
            why = (f"{n['name']} has {n['mem_available'] / GiB:.1f} GiB available and changing the experts needs "
                   f"{need / GiB:.1f} GiB on every node (layer {layer}: its new slabs {slabs / GiB:.1f} GiB after "
                   f"{grown / GiB:.1f} GiB grown in the layers before it, and {self.swap_margin // GiB} GiB to spare): "
                   "a start with more kept for the OS leaves room for it (a panel's start gives the KV all the room its plan "
                   "leaves, whatever the context length; a start from glm53.conf alone: a smaller KV_SIZE)")
        return {"need_mib": need >> 20, "margin_mib": self.swap_margin >> 20, "layer": layer, "layer_slabs_mib": slabs >> 20,
                "grown_before_mib": grown >> 20, "ranks": ranks, "fits": not short, "why": why}

    def snapshot(self, label):
        """LAUNCH_SNAPSHOTS=1: each node's GPU, memory, containers and log tail before and after a start or a stop, in
        PKG/snapshots/<time>-<what>/ (off by default)."""
        if not self.snapshots:
            return
        d = os.path.join(self.pkg_local, "snapshots", label)
        os.makedirs(d, exist_ok=True)
        cmd = ("echo '## nvidia-smi'; nvidia-smi; echo '## free -h'; free -h; echo '## docker ps'; docker ps 2>&1; "
               f"echo '## log tail'; for s in {' '.join(f"rank{r}" for r in range(len(self.nodes)))}; do {self.keep} cat $s 2>/dev/null | tail -n 5; done")
        for name, (rc, out) in zip(self.names, self.on_all(cmd, 60)):
            open(os.path.join(d, f"{name}.txt"), "w").write(out)
        h = "200" if self.health() else "down"
        open(os.path.join(d, "health.txt"), "w").write(f"{self.engine[0]}:{self.engine[1]}/health {h}\n")


# ---- the servers and the ring (R-017): the panel's first step and the live panel's [ RING ] ----
# Only what the head reaches over ConnectX-7 cables (LLDP) is looked for; the LAN is not scanned and other heads are
# not asked. Read by the launcher on the head over ssh, at the hosts' LAN addresses: its own key, never a password, a host key it does not know never added
# (StrictHostKeyChecking=yes: the owner adds host keys). The ring's link subnets are never scanned and never used for ssh.
SSH_STRICT = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=5"]
# one host: key=value lines; "dev=" a CX7 half (device, interface, operstate, IPv4 address or -, firmware, the index
# of its RoCE v2 GID with an IPv4 address or -); its LLDP neighbours (a neighbour's lines start with .via)
HOST_PROBE = r'''echo "host=$(hostname)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | sed 's|.*/||' | sort -u | tr '\n' ' ')"
awk '/^MemTotal:/ {print "mem_total_kb=" $2} /^MemAvailable:/ {print "mem_available_kb=" $2}' /proc/meminfo
echo "kernel=$(uname -r)"
echo "driver=$(cat /sys/module/nvidia/version 2>/dev/null)"
for d in /sys/class/infiniband/roce*; do
  [ -d "$d" ] || continue; h=${d##*/}; n=$(ls $d/device/net 2>/dev/null | head -1); [ -n "$n" ] || continue
  g=-; for i in $(seq 0 15); do t=$(cat $d/ports/1/gid_attrs/types/$i 2>/dev/null); a=$(cat $d/ports/1/gids/$i 2>/dev/null)
    case "$t:$a" in 'RoCE v2:0000:0000:0000:0000:0000:ffff:'*) g=$i ;; esac; done
  echo "dev=$h $n $(cat /sys/class/net/$n/operstate) $(ip -4 -o addr show dev $n | awk '{print $4; exit}' | grep . || echo -) $(cat $d/fw_ver | tr ' ' _) $g"
done
lldpctl -f keyvalue 2>/dev/null | grep -E '^lldp\.[^.]+\.(via|chassis\.name|chassis\.mac|port\.descr)='
'''
# a ConnectX-7 net device (enp1s0f1np1, enP2p1s0f0np0), as tools/topology.py reads them: SOCKET_IFNAME naming one is
# the fabric, not the management network the hosts are read at
CX7_NET = re.compile(r"^en[pP].*s0f[01]np[01]$")
# the node check's discovery before one ran (GET /v1/launch/discover)
DISC_IDLE = {"state": "idle", "started_unix": 0, "ended_unix": 0, "kind": "", "why": "", "head": None, "subnets": [],
             "fabric": None, "hosts": [], "swept": False}
SWEEP_TIMEOUT = 0.4  # s a host on a CX7 /24 has to accept ssh's port (tcp/22) in the discovery's sweep
SWEEP_AT_ONCE = 128


def default_route():
    """The default route's interface and this node's IPv4 address on it (the management network), ("", "") if none."""
    try:
        out = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True, timeout=5).stdout
        f = out.split()
        dev = f[f.index("dev") + 1] if "dev" in f else ""
        a = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", dev], capture_output=True, text=True, timeout=5).stdout.split() if dev else []
        return dev, a[a.index("inet") + 1].split("/")[0] if "inet" in a else ""
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "", ""


def answers_ssh(addr):
    """Whether addr accepts a connection on tcp/22 within SWEEP_TIMEOUT (nothing is sent)."""
    try:
        with socket.create_connection((addr, 22), timeout=SWEEP_TIMEOUT):
            return True
    except OSError:
        return False


# each lane's counters on a ring node: device, interface, operstate, IPv4 address or -, port_xmit_data (4-byte units)
LINK_PROBE = r'''for h in %s; do n=$(ls /sys/class/infiniband/$h/device/net/ 2>/dev/null | head -1); [ -n "$n" ] || continue
  echo "$h $n $(cat /sys/class/net/$n/operstate) $(ip -4 -o addr show dev $n | awk '{print $4; exit}' | grep . || echo -) $(cat /sys/class/infiniband/$h/ports/1/counters/port_xmit_data)"; done'''
LINKS_EVERY = 2.0  # s between two readings of the lanes' counters
LINKS_IDLE = 15.0  # s after the last GET /v1/launch/links the readings stop
BW_PORT = 18531  # ib_write_bw's port for the lanes' measurement (run i of a wave: BW_PORT + i)
BW_SECONDS = 3
BW_WAVE_S = BW_SECONDS + 3.2  # a wave's seconds before one has run: ib_write_bw's -D, the server's 1.5 s head start, ssh
# which runs go at once (bw_waves; TF_LAUNCHER_BW overrides). A node of the ring sends
# and receives about 196 Gb/s at most however its lanes share it, so the more at once, the less a lane reads (10-31,
# 16 runs: one 98 s 109.1-109.3 Gb/s; nodes 49 s 109.1-109.3; duplex 25 s 106.5-107.1; half 13 s 98.0; all 16 at once
# 6 s 48, not offered)
BW_TOGETHER = os.environ.get("TF_LAUNCHER_BW", "half")   # 13 s on the ring


def cx7_port(name):
    """A CX7 device's or interface's port (rocep1s0f1, enp1s0f1np1: 1), None for another name."""
    m = re.search(r"s0f(\d)", name)
    return int(m.group(1)) if m else None


def cx7_half(name):
    """A CX7 device's or interface's PCIe half: 1 for the second (roceP2p1s0f0, enP2p1s0f0np0), else 0."""
    return 1 if re.search(r"P2p", name) else 0


def ssh_access(err):
    """Why ssh did not run the command (its own error text): hostkey, nokey or noanswer."""
    if re.search(r"Host key verification failed|IDENTIFICATION HAS CHANGED|No \S+ host key is known", err):
        return "hostkey"
    if "Permission denied" in err:
        return "nokey"
    return "noanswer"


def bw_waves(runs, together="duplex"):
    """The runs' indexes in waves, taken half by half and direction by direction, each in the first wave it fits.
    one: a run a wave. nodes: no two runs of a wave share a node. duplex: a node sends at most once and receives at most
    once in a wave (a ring of four: every node to its next, 4 waves). half: the same, a lane half at a time (2 waves)."""
    waves = []
    for i in sorted(range(len(runs)), key=lambda i: (runs[i][1], runs[i][2])):
        lk, h, d = runs[i]
        src, dst = (lk["a"], lk["b"]) if d == "ab" else (lk["b"], lk["a"])
        ends = ({"one"} if together == "one" else {lk["a"], lk["b"]} if together == "nodes"
                else {(src, h, "out"), (dst, h, "in")} if together == "half" else {(src, "out"), (dst, "in")})
        for w in waves:
            if not ends & w[1]:
                w[0].append(i); w[1].update(ends)
                break
        else:
            waves.append(([i], set(ends)))
    return [w[0] for w in waves]


def bw_gbps(out):
    """ib_write_bw's average in Gb/s (--report_gbits): the last row of its table, #bytes #iterations peak average rate."""
    v = None
    for line in out.splitlines():
        m = re.match(r"\s*\d+\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$", line)
        if m:
            v = float(m.group(1))
    return v


class Servers:
    """The hosts the head reaches over ConnectX-7 cables (a scan when the panel asks) or one host by its LAN address, the
    ring traced from the head over LLDP, each lane's traffic and the lanes' bandwidth (R-017). Nothing here starts or
    stops an engine; bw and pull run only while this lane's engine is stopped."""

    def __init__(self, L, ifname, hcas):
        self.L = L
        self.ifname = ifname
        self.hcas = hcas
        self.lock = threading.Lock()
        self.hosts = {}  # addr -> host (scan_view's hosts[], with _mac and ports[]._peer_mac kept inside)
        self.scan = {"state": "idle", "started_unix": 0, "ended_unix": 0, "unknown": []}
        self.ring = {"state": "idle", "ring": "none", "nodes": [], "names": [], "links": [], "conf_match": False,
                     "problems": [], "traced_unix": 0}
        self.links, self.links_t, self.links_asked, self.links_run = [], 0.0, 0.0, False
        self.bw = {"state": "idle", "started_unix": 0, "ended_unix": 0, "lanes": [], "why": ""}
        self.pull = {"state": "idle", "nodes": []}
        self.disc = dict(DISC_IDLE)  # the node check's discovery (POST /v1/launch/discover): the last one

    # ---- the LAN and one host ----
    def lan_ifname(self):
        """The interface of the LAN the hosts are read at: SOCKET_IFNAME when it names the management network (a
        ring's), else (unset, auto, or a ConnectX-7 net device: a switch's fabric, or a conf not set yet) the default
        route's interface."""
        if self.ifname and self.ifname != "auto" and not CX7_NET.match(self.ifname):
            return self.ifname
        return default_route()[0] or self.ifname

    def lan(self):
        """The head's address on the LAN interface (lan_ifname) and its network: the LAN the hosts are read at. The
        network only says which addresses are on the LAN: nothing sweeps it, whatever its size."""
        ifn = self.lan_ifname()
        if not ifn:
            raise Fail("no LAN interface: SOCKET_IFNAME is not set and there is no default route")
        try:
            out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", ifn], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        for line in out.splitlines():
            f = line.split()
            if "inet" in f:
                i = ipaddress.ip_interface(f[f.index("inet") + 1])
                return str(i.ip), i.network
        raise Fail(f"{ifn}: no IPv4 address on this node")

    def rname(self, addr):
        """A host's name without ssh: the resolver's reverse name, "" when none."""
        try:
            out = subprocess.run(["getent", "hosts", addr], capture_output=True, text=True, timeout=2).stdout.split()
            return out[1].split(".")[0] if len(out) > 1 else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def probe_host(self, addr, local=False):
        """One host read over ssh: access, the GPU, the memory, the four CX7 halves and their LLDP neighbours, the
        versions (launch.probe). local: this node, read without ssh (the discovery's head)."""
        h = {"addr": addr, "name": "", "access": "noanswer", "gb10": None, "gpu": "", "other": "", "mem_total": 0,
             "mem_available": 0, "ports": [], "versions": {}, "probed_unix": int(time.time()), "_gids": {}}
        try:
            r = subprocess.run((["bash", "-c"] if local else SSH_STRICT + [addr]) + [HOST_PROBE], capture_output=True, text=True, timeout=25)
        except subprocess.TimeoutExpired:
            r = None
        if r is None or r.returncode == 255:
            h["access"] = ssh_access(r.stderr if r else "")
            h["name"] = self.rname(addr)
            return h
        kv, devs, nb = {}, [], {}
        for line in r.stdout.splitlines():
            if line.startswith("lldp."):
                k, _, v = line.partition("=")
                p = k.split(".", 2)
                if len(p) < 3:
                    continue
                if p[2] == "via":
                    nb.setdefault(p[1], []).append({})
                elif nb.get(p[1]):
                    nb[p[1]][-1][{"chassis.name": "name", "chassis.mac": "mac", "port.descr": "net"}[p[2]]] = v
            elif line.startswith("dev="):
                devs.append(line[4:].split())
            else:
                k, _, v = line.partition("=")
                kv[k] = v.strip()
        name = kv.get("host", "")
        h.update(access="ok", name=name, gpu=kv.get("gpu", ""), other=", ".join(kv.get("apps", "").split()))
        h["gb10"] = "GB10" in h["gpu"]
        try:
            h["mem_total"] = int(kv.get("mem_total_kb", 0)) * 1024
            h["mem_available"] = int(kv.get("mem_available_kb", 0)) * 1024
        except ValueError:
            pass
        fw = {}
        for d in devs:
            if len(d) < 6 or cx7_port(d[0]) is None:
                continue
            dev, net, state, a4, fwv, gid = d[:6]
            fw[dev] = fwv.replace("_", " ")
            # a neighbour that is this host is the port's other half: dropped; the one on the same half preferred
            peers = [x for x in nb.get(net, []) if x.get("name") and x.get("name") != name]
            pick = next((x for x in peers if cx7_half(x.get("net", "")) == cx7_half(net)), peers[0] if peers else None)
            peer = ({"name": pick["name"], "net": pick.get("net", ""), "port": cx7_port(pick.get("net", "")),
                     "half": cx7_half(pick.get("net", ""))} if pick else None)
            h["ports"].append({"dev": dev, "net": net, "port": cx7_port(dev), "half": cx7_half(dev), "up": state == "up",
                               "addr": "" if a4 == "-" else a4, "peer": peer, "_peer_mac": (pick or {}).get("mac", "")})
            if gid != "-":
                h["_gids"][dev] = int(gid)
        h["ports"].sort(key=lambda p: (p["port"], p["half"]))
        if not h["gb10"]:
            h["ports"] = []
        h["versions"] = {"kernel": kv.get("kernel", ""), "driver": kv.get("driver", ""),
                         "cx7_fw": fw.get("rocep1s0f0") or (fw[sorted(fw)[0]] if fw else "")}
        return h

    def resolve(self, a):
        """An address or a name to an address of the head's LAN (launch.probe's addr)."""
        _, net = self.lan()
        try:
            ip = ipaddress.ip_address(a)
            if ip not in net:
                raise Fail(f"addr: {a} is not on the head's LAN ({net})")
            return a
        except ValueError:
            pass
        try:
            found = {i[4][0] for i in socket.getaddrinfo(a, None, socket.AF_INET)}
        except OSError:
            found = set()
        for x in sorted(found):
            if ipaddress.ip_address(x) in net:
                return x
        raise Fail(f"addr: {a} has no address on the head's LAN ({net})" if found else f"addr: {a} is not known")

    def view(self, h):
        """A host as the API gives it: the inside fields dropped."""
        v = {k: x for k, x in h.items() if not k.startswith("_")}
        v["ports"] = [{k: x for k, x in p.items() if not k.startswith("_")} for p in h["ports"]]
        return v

    def probe_req(self, body):
        """POST /v1/launch/probe {"addr"}: one host now; kept with the scan's hosts."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        a = j.get("addr") if isinstance(j, dict) else None
        if not isinstance(a, str) or not ADDR.fullmatch(a.strip()):
            raise Fail("addr: an address or a name")
        h = self.probe_host(self.resolve(a.strip()))
        with self.lock:
            if h["access"] != "noanswer":  # a host that answered is kept with the scan's
                self.hosts[h["addr"]] = h
            return 200, self.view(h)

    # ---- the scan ----
    def begin_scan(self):
        """POST /v1/launch/scan: from the head, every host reached over ConnectX-7 cables (each port's LLDP neighbour,
        then theirs), each read over ssh at its LAN address; names with no LAN address known go to unknown."""
        self.lan()
        with self.lock:
            if self.scan["state"] == "scanning":
                return 409, {"error": "a scan runs"}
            self.scan = {"state": "scanning", "started_unix": int(time.time()), "ended_unix": 0, "unknown": []}
        threading.Thread(target=self.run_scan, daemon=True).start()
        return 202, {"state": "scanning"}

    def run_scan(self):
        found, unknown = {}, []
        try:
            head, net = self.lan()
            todo = [head]
            while todo and len(found) < 16:
                with ThreadPoolExecutor(len(todo)) as ex:
                    read = list(ex.map(self.probe_host, todo))
                todo = []
                for h in read:
                    found[h["addr"]] = h
                for h in read:
                    for p in h["ports"]:
                        n = p["peer"]["name"] if p["peer"] else ""
                        if not n or n in unknown or any(x["name"] == n for x in found.values()):
                            continue
                        a = self.addr_of(n, p["_peer_mac"], net)
                        if not a:
                            unknown.append(n)
                        elif a not in found and a not in todo:
                            todo.append(a)
            log(f"scan over ConnectX-7: {' '.join(h['name'] or h['addr'] for h in found.values())}" + (f"; no LAN address for {' '.join(unknown)}" if unknown else ""))
        except Exception as e:
            log(f"scan: {e!r}")
        finally:
            with self.lock:
                if found:
                    self.hosts = found
                self.scan.update(state="done", ended_unix=int(time.time()), unknown=sorted(unknown))

    def scan_view(self):
        with self.lock:
            hosts = sorted(self.hosts.values(), key=lambda h: ipaddress.ip_address(h["addr"]))
            return dict(self.scan, hosts=[self.view(h) for h in hosts])

    # ---- the node check's discovery (the panel's first screen: a ring or a switch) ----
    def begin_discover(self):
        """POST /v1/launch/discover: how the head is cabled and, on a switch, the hosts to choose from, in the
        background; read-only on every node."""
        if self.L.check_view()["state"] == "checking":
            return 409, {"error": "a check is running"}
        with self.lock:
            if self.disc["state"] == "discovering":
                return 409, {"error": "a discovery runs"}
            self.disc = dict(DISC_IDLE, state="discovering", started_unix=int(time.time()))
        threading.Thread(target=self.run_discover, daemon=True).start()
        return 202, {"state": "discovering"}

    def run_discover(self):
        try:
            res = self.discover()
        except Exception as e:
            log(f"discover: {e!r}")
            res = {"kind": "none", "why": str(e)}
        with self.lock:
            self.disc = dict(self.disc, **res, state="done", ended_unix=int(time.time()))
            d = self.disc
        log(f"discover: {d['kind']}" + (f" on {d['fabric']['subnet']}, {len(d['hosts'])} hosts" if d.get("fabric") else "")
            + (f" ({d['why']})" if d.get("why") else ""))

    def discover(self):
        """The head read here (its CX7 halves, their LLDP neighbours) and its management address; then:
        ring    LLDP names GB10 neighbours, two or more, on both ports (nothing swept: a ring's link subnets are not
                scanned), or, LLDP silent, one other host on a /24 of port 0 and one on a /24 of port 1
        switch  a /24 of a CX7 half with two or more other hosts that accept ssh's port (tcp/22): the fabric, the /24
                with the most (a tie: IB_HCA's device's, then the first half), every host on it read over ssh there
        direct  one other host only (one cable): as a switch, that host to choose
        none    no other host answers over CX7
        Only the head's own CX7 /24s are swept (connect only); the management LAN is never swept."""
        ifn, maddr = default_route()
        if self.ifname and self.ifname != "auto" and not CX7_NET.match(self.ifname):
            try:
                maddr, _ = self.lan()
                ifn = self.ifname
            except Fail:
                pass
        head = self.probe_host(maddr or "127.0.0.1", local=True)
        name = head["name"] or socket.gethostname()

        def ip_of(p):
            return p["addr"].split("/")[0] if p["addr"] else ""

        def net_of(p):
            return ipaddress.ip_network(ip_of(p) + "/24", strict=False) if p["addr"] else None

        ports = sorted(head["ports"], key=lambda p: (p["port"], p["half"]))
        hp = [{"dev": p["dev"], "net": p["net"], "port": f"p{p['port']}", "half": p["half"], "up": p["up"], "addr": ip_of(p),
               "subnet": str(net_of(p)) if p["addr"] else "", "others": 0, "peer": p["peer"]["name"] if p["peer"] else ""} for p in ports]
        out = {"head": {"name": name, "lan": {"ifname": ifn, "addr": maddr}, "ports": hp}, "subnets": [], "fabric": None,
               "hosts": [], "swept": False, "kind": "none", "why": ""}
        if head["access"] != "ok" or head["gb10"] is False:
            out["why"] = "this node is not a GB10 (nvidia-smi names no GB10)" if head["access"] == "ok" else "this node could not be read"
            return out
        gb = [p for p in ports if p["up"] and p["peer"] and CX7_NET.match(p["peer"]["net"] or "")]
        if len({p["peer"]["name"] for p in gb}) >= 2 and len({p["port"] for p in gb}) >= 2:
            out["kind"] = "ring"
            return out
        # the head's CX7 /24s, the first half on each in port, half order; its own addresses left out of the sweep
        subs = {}
        for p in ports:
            if p["up"] and p["addr"] and net_of(p) not in subs:
                subs[net_of(p)] = p
        if not subs:
            out["why"] = "no ConnectX-7 half of this node is up with an IPv4 address"
            return out
        mine = self.L.own_addrs()
        targets = [str(a) for n in subs for a in n.hosts() if str(a) not in mine]
        with ThreadPoolExecutor(SWEEP_AT_ONCE) as ex:
            hit = {a for a, ok in zip(targets, ex.map(answers_ssh, targets)) if ok}
        out["swept"] = True
        key = lambda a: ipaddress.ip_address(a)  # noqa: E731
        out["subnets"] = [{"subnet": str(n), "dev": p["dev"], "net": p["net"], "head_addr": ip_of(p),
                           "others": sorted((a for a in hit if ipaddress.ip_address(a) in n), key=key), "_port": p["port"]}
                          for n, p in subs.items()]
        for x in hp:
            x["others"] = next((len(s["others"]) for s in out["subnets"] if s["subnet"] == x["subnet"]), 0)
        wide = [s for s in out["subnets"] if len(s["others"]) >= 2]
        ones = [s for s in out["subnets"] if len(s["others"]) == 1]
        pref = self.hcas[0] if self.hcas else ""

        def pick(cands):
            most = max(len(s["others"]) for s in cands)
            top = [s for s in cands if len(s["others"]) == most]
            return next((s for s in top if s["dev"] == pref), top[0])

        if wide:
            out["kind"], fab = "switch", pick(wide)
        elif len({s["_port"] for s in ones}) >= 2:
            out["kind"], fab = "ring", None
        elif ones:
            out["kind"], fab = "direct", pick(ones)
        else:
            out["why"] = "no other host accepts ssh on this node's ConnectX-7 /24s"
            fab = None
        for s in out["subnets"]:
            s.pop("_port", None)
        if fab:
            out["fabric"] = {k: fab[k] for k in ("subnet", "dev", "net", "head_addr")}
            others = fab["others"][:32]
            with ThreadPoolExecutor(max(1, len(others))) as ex:
                read = list(ex.map(self.probe_host, others))
            with self.L.lock:
                lane = set(self.L.nodes)
            out["hosts"] = [dict(self.view(h), in_lane=h["addr"] in lane) for h in read]
        return out

    def disc_view(self):
        with self.lock:
            return json.loads(json.dumps(self.disc))

    # ---- the ring ----
    def addr_of(self, name, mac, net):
        """The LAN address of a neighbour LLDP names: a host read before, a node of NODES, the head's ARP table (the
        neighbour's chassis MAC is its LAN interface's), or the resolver; "" when none is on the LAN."""
        with self.lock:
            for h in self.hosts.values():
                if h["name"] == name and h["access"] == "ok":
                    return h["addr"]
        with self.L.lock:
            for n, a in zip(self.L.names, self.L.nodes):
                if n == name:
                    try:
                        if ipaddress.ip_address(a) in net:
                            return a
                    except ValueError:
                        pass
        if mac:
            try:
                out = subprocess.run(["ip", "neigh", "show", "dev", self.lan_ifname()], capture_output=True, text=True, timeout=5).stdout
            except (OSError, subprocess.TimeoutExpired):
                out = ""
            for line in out.splitlines():
                f = line.split()
                if "lladdr" in f and f[f.index("lladdr") + 1].lower() == mac.lower() and "FAILED" not in f:
                    return f[0]
        for n in (name, name + ".local"):
            try:
                for i in socket.getaddrinfo(n, None, socket.AF_INET):
                    if ipaddress.ip_address(i[4][0]) in net:
                        return i[4][0]
            except OSError:
                pass
        return ""

    def begin_ring(self):
        """POST /v1/launch/ring: the ring traced again from the head."""
        with self.lock:
            if self.ring["state"] == "tracing":
                return 409, {"error": "a trace runs"}
            self.ring = dict(self.ring, state="tracing")
        threading.Thread(target=self.run_ring, daemon=True).start()
        return 202, {"state": "tracing"}

    def run_ring(self):
        try:
            res = self.trace()
        except Exception as e:
            log(f"ring trace: {e!r}")
            res = {"ring": "none", "nodes": [], "names": [], "links": [], "conf_match": False,
                   "problems": [{"addr": "", "name": "", "what": "noanswer", "detail": str(e)}]}
        with self.lock:
            self.ring = dict(res, state="done", traced_unix=int(time.time()))
        log(f"ring: {res['ring']}, {' '.join(res['names'])}" + (f"; {len(res['problems'])} problems" if res["problems"] else ""))

    def trace(self):
        """Out of the head's port 0, which arrives at the next node's port 1 (the crossed cabling), out of that node's
        other port, and so on until back at the head, a node that cannot be read, or 8 hops; when the ring is open, out
        of the head's port 1 the other way too. Each node read is kept with the scan's hosts."""
        head_addr, net = self.lan()
        order, links, problems, seen = [], [], [], {}

        def read(addr):
            h = self.probe_host(addr)
            with self.lock:
                self.hosts[addr] = h
            return h

        def problem(h, what, detail=""):
            problems.append({"addr": h["addr"], "name": h["name"], "what": what, "detail": detail})

        head = read(head_addr)
        order.append(head)
        seen[head["addr"]] = 0
        if head["access"] != "ok" or not head["ports"] or not any(p["peer"] for p in head["ports"]):
            if head["access"] != "ok":
                problem(head, head["access"])
            else:
                problem(head, "open", "no GB10 neighbour on port 0 or port 1")
            return {"ring": "none", "nodes": [head_addr], "names": [head["name"]], "links": [], "conf_match": False,
                    "problems": problems}

        def walk(out, forward):
            cur = head
            for _ in range(8):
                ps = [p for p in cur["ports"] if p["port"] == out and p["peer"]]
                if not ps:
                    problem(cur, "open", f"port {out}")
                    return False
                pn, arr, mac = ps[0]["peer"]["name"], ps[0]["peer"]["port"], ps[0]["_peer_mac"]
                if pn == head["name"]:
                    nxt = head
                else:
                    a = self.addr_of(pn, mac, net)
                    if not a:
                        # listed by name with no address (the panel offers to add it by its address); no link to it
                        if not any(h["name"] == pn and not h["addr"] for h in order):
                            order.append({"addr": "", "name": pn, "other": ""})
                        problems.append({"addr": "", "name": pn, "what": "noaddr", "detail": "no LAN address known: add it by its address"})
                        return False
                    nxt = order[seen[a]] if a in seen else read(a)
                    if nxt["access"] != "ok":
                        problem(nxt, nxt["access"])
                        return False
                    if not nxt["gb10"]:
                        problem(nxt, "not_gb10")
                        return False
                if nxt is not head and nxt["addr"] not in seen:
                    seen[nxt["addr"]] = len(order)
                    order.append(nxt)
                a_r, b_r = seen[cur["addr"]], seen[nxt["addr"]]
                links.append(self.link(cur, out, nxt, arr, a_r if forward else b_r, b_r if forward else a_r, problem))
                if nxt is head:
                    return True
                if not forward and b_r < len(order) - 1:  # the other way round, back at a node reached forward
                    return False
                cur, out = nxt, 1 - arr
            problem(cur, "open", "8 hops without coming back to the head")
            return False

        closed = walk(0, True)
        if not closed:
            walk(1, False)
        for h in order:
            if h["other"] and not self.ours(h):
                problem(h, "busy", h["other"])
        nodes = [h["addr"] for h in order]
        problems = [dict(t) for t in dict.fromkeys(tuple(p.items()) for p in problems)]
        return {"ring": "closed" if closed else "open", "nodes": nodes, "names": [h["name"] for h in order],
                "links": links, "conf_match": closed and nodes == list(self.L.nodes), "problems": problems}

    def link(self, cur, out, nxt, arr, a, b, problem):
        """A cable from cur's port `out` to nxt's port `arr`: each half's lane up on both ends and its /24; the devices
        and GIDs kept inside for bw."""
        lanes, subnets, devs = [False, False], ["", ""], [None, None]
        for p in cur["ports"]:
            if p["port"] != out or not p["peer"]:
                continue
            q = next((x for x in nxt["ports"] if x["net"] == p["peer"]["net"]), None)
            h = p["half"]
            lanes[h] = bool(p["up"] and q and q["up"])
            subnets[h] = str(ipaddress.ip_interface(p["addr"]).network) if p["addr"] else ""
            if q:
                devs[h] = {cur["addr"]: (p["dev"], cur["_gids"].get(p["dev"])), nxt["addr"]: (q["dev"], nxt["_gids"].get(q["dev"]))}
        crossed = out != arr
        if not crossed:
            problem(cur, "uncrossed", nxt["name"])
        for h in (0, 1):
            if not lanes[h]:
                problem(cur, "half", f"half {h}")
        return {"a": a, "b": b, "lanes": lanes, "subnets": subnets, "crossed": crossed, "_devs": devs}

    def ours(self, h):
        """The GPU processes of a node of this lane while its engine runs are this lane's."""
        return self.L.engine_up and h["addr"] in self.L.nodes and set(h["other"].split(", ")) <= {"tensorfold-native", "tensorfold-nati"}

    def ring_view(self):
        with self.lock:
            r = dict(self.ring)
        r["links"] = [{k: x for k, x in lk.items() if not k.startswith("_")} for lk in r["links"]]
        return r

    # ---- each lane's traffic ----
    def links_view(self):
        """GET /v1/launch/links: the lanes' last readings; a reader thread runs while the panel asks."""
        if self.L.topology != "ring4":
            return {"age_ms": 0, "links": []}
        with self.lock:
            self.links_asked = time.time()
            start = not self.links_run
            self.links_run = True
            v = {"age_ms": int((time.time() - self.links_t) * 1000) if self.links_t else 0, "links": [dict(x) for x in self.links]}
        if start:
            threading.Thread(target=self.read_links, daemon=True).start()
        return v

    def read_links(self):
        cmd = LINK_PROBE % " ".join(self.hcas)
        prev, t_prev = None, 0.0
        try:
            while time.time() - self.links_asked < LINKS_IDLE:
                t0 = time.time()
                rows = []
                for rc, out in self.L.on_all(cmd, 10):
                    d = {}
                    for line in out.splitlines():
                        f = line.split()
                        if len(f) == 5:
                            d[f[0]] = {"up": f[2] == "up", "net": None if f[3] == "-" else ipaddress.ip_interface(f[3]).network,
                                       "x": int(f[4]) if f[4].isdigit() else None}
                    rows.append(d if rc == 0 else None)
                t = time.time()
                lanes = []
                w = len(rows)
                for r in range(w if w == 4 else 0):
                    s = (r + 1) % w
                    if rows[r] is None or rows[s] is None:
                        continue
                    for dr, xr in sorted(rows[r].items(), key=lambda kv: cx7_half(kv[0])):
                        ds = next((k for k, x in rows[s].items() if xr["net"] and x["net"] == xr["net"]), None)
                        if ds is None:
                            continue
                        xs = rows[s][ds]
                        up = xr["up"] and xs["up"]
                        g = None
                        if up and prev and prev[r] and prev[s] and dr in prev[r] and ds in prev[s]:
                            d = [(xr["x"], prev[r][dr]["x"]), (xs["x"], prev[s][ds]["x"])]
                            if all(n is not None and o is not None and n >= o for n, o in d):
                                g = round(sum(n - o for n, o in d) * 32 / 1e9 / max(t - t_prev, 1e-3), 2)
                        lanes.append({"a": r, "b": s, "half": cx7_half(dr), "up": up, "gbps": g if up else None})
                with self.lock:
                    self.links, self.links_t = lanes, t
                prev, t_prev = rows, t
                time.sleep(max(0.0, LINKS_EVERY - (time.time() - t0)))
        finally:
            with self.lock:
                self.links_run = False

    # ---- the lanes' bandwidth ----
    def begin_bw(self):
        """POST /v1/launch/bw: ib_write_bw on every lane of the traced ring, each way, in the waves of bw_waves
        (BW_TOGETHER)."""
        if not self.L.stopped():
            return 409, {"error": "the engine is not stopped"}
        with self.lock:
            if self.bw["state"] == "measuring":
                return 409, {"error": "a measurement runs"}
            ring = self.ring
            if ring.get("ring") != "closed":
                return 412, {"error": "the ring is not closed: trace it (POST /v1/launch/ring)"}
            runs = [(lk, h, d) for lk in ring["links"] for h in (0, 1) for d in ("ab", "ba")]
            self.bw = {"state": "measuring", "started_unix": int(time.time()), "ended_unix": 0, "why": "",
                       "lanes": [{"a": lk["a"], "b": lk["b"], "half": h, "dir": d, "gbps": None} for lk, h, d in runs]}
        threading.Thread(target=self.run_bw, args=(ring, runs), daemon=True).start()
        return 202, {"state": "measuring"}

    def gpu_busy(self, nodes):
        """The nodes running a GPU program (nvidia-smi's compute apps), or not answering."""
        def one(n):
            try:
                r = subprocess.run(SSH_STRICT + [n, "nvidia-smi --query-compute-apps=pid --format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20)
                return n if r.returncode != 0 or r.stdout.strip() else None
            except subprocess.TimeoutExpired:
                return n
        with ThreadPoolExecutor(len(nodes)) as ex:
            return [n for n in ex.map(one, nodes) if n]

    def run_bw(self, ring, runs):
        nodes, why = ring["nodes"], ""
        try:
            # stopped() sees this lane's engine only: another lane's engine on the ring shares the links, and the runs
            # would disturb it (a check-only launcher beside a running lane measured nothing but would have run them)
            busy = self.gpu_busy(nodes)
            if busy:
                why = f"a GPU program runs on {', '.join(busy)} (another engine uses the links): nothing measured"
            # the runs in waves (BW_TOGETHER); eta_unix: when the last wave should end, at the pace of the waves so far
            # (BW_WAVE_S before one ended), so the page counts down
            waves = [] if busy else bw_waves(runs, BW_TOGETHER)
            t0 = time.time()
            with self.lock:
                self.bw.update(waves=len(waves), eta_unix=round(t0 + len(waves) * BW_WAVE_S, 1))
            for k, wave in enumerate(waves):
                if not self.L.stopped():
                    why = "the engine started: the measurement stopped"
                    break
                with ThreadPoolExecutor(len(wave)) as ex:
                    list(ex.map(lambda i: self.bw_one(nodes, i, *runs[i], BW_PORT + wave.index(i)), wave))
                with self.lock:
                    self.bw["eta_unix"] = round(time.time() + (len(waves) - k - 1) * (time.time() - t0) / (k + 1), 1)
            log("bw: " + ", ".join(f"{x['a']}-{x['b']}.{x['half']}{x['dir']} {x['gbps']}" for x in self.bw["lanes"]))
        except Exception as e:
            why = repr(e)
        with self.lock:
            done = any(x["gbps"] is not None for x in self.bw["lanes"])
            self.bw.update(state="failed" if why or not done else "done", ended_unix=int(time.time()),
                           why=why or ("" if done else "no lane measured"))

    def bw_one(self, nodes, i, lk, h, d, port):
        """One run: ib_write_bw from one end of lane (lk, h) to the other (d: ab or ba) on `port`; its Gb/s into lanes[i]."""
        devs = lk["_devs"][h]
        if not devs:
            return
        src, dst = (nodes[lk["a"]], nodes[lk["b"]]) if d == "ab" else (nodes[lk["b"]], nodes[lk["a"]])
        (sd, sg), (dd, dg) = devs[src], devs[dst]
        if sg is None or dg is None:
            return
        arg = f"-d {sd} -x {sg} -F --report_gbits -D {BW_SECONDS} -p {port}"
        server = subprocess.Popen(SSH_STRICT + [dst, f"timeout 30 ib_write_bw -d {dd} -x {dg} -F --report_gbits -D {BW_SECONDS} -p {port}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        try:
            out = subprocess.run(SSH_STRICT + [src, f"timeout 30 ib_write_bw {arg} {dst}"], capture_output=True, text=True, timeout=40).stdout
        except subprocess.TimeoutExpired:
            out = ""
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        g = bw_gbps(out)
        with self.lock:
            self.bw["lanes"][i]["gbps"] = g

    def bw_view(self):
        """GET /v1/launch/bw; now_unix: this host's clock, which eta_unix and started_unix read by"""
        with self.lock:
            return dict(self.bw, lanes=[dict(x) for x in self.bw["lanes"]], now_unix=round(time.time(), 1))

    # ---- the image ----
    def begin_pull(self, body):
        """POST /v1/launch/pull {"nodes"?}: tf-glm53 pull on the head and those nodes (every node when absent)."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        want = j.get("nodes") if isinstance(j, dict) else None
        nodes = list(self.L.nodes)
        if want is not None:
            if not isinstance(want, list) or not all(isinstance(x, str) and x in nodes for x in want):
                raise Fail("nodes: addresses of NODES")
            nodes = [nodes[0]] + [x for x in nodes[1:] if x in want]
        if self.L.runtime != "docker":
            return 412, {"error": "RUNTIME is host: there is no image to pull"}
        if not self.L.stopped():
            return 409, {"error": "the engine is not stopped"}
        with self.lock:
            if self.pull["state"] == "pulling":
                return 409, {"error": "a pull runs"}
            self.pull = {"state": "pulling", "nodes": [{"addr": a, "state": "pulling", "why": ""} for a in nodes]}
        threading.Thread(target=self.run_pull, args=(nodes,), daemon=True).start()
        return 202, {"state": "pulling"}

    def run_pull(self, nodes):
        # the lane's node file with NODES these nodes, read after CONF in place of it (NODES_CONF)
        tmp = os.path.join(self.L.dir, "pull-" + time.strftime("%Y%m%d-%H%M%S") + ".nodes.conf")
        rc, out = 1, ""
        try:
            try:
                lane = open(self.L.node_file).read() if self.L.node_file else ""
            except OSError:
                lane = ""
            open(tmp, "w").write(lane + ("" if not lane or lane.endswith("\n") else "\n") + "NODES=(" + " ".join(nodes) + ")\n")
            p = subprocess.run(["bash", os.path.join(self.L.pkg_local, "tf-glm53"), "-c", self.L.conf, "pull"], capture_output=True, text=True,
                               timeout=3600, env=dict(os.environ, NODES_CONF=tmp))
            rc, out = p.returncode, p.stdout + p.stderr
        except (OSError, subprocess.TimeoutExpired, Fail) as e:
            out = str(e)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        last = " ".join(out.strip().splitlines()[-2:])[-300:]
        with self.lock:
            for n in self.pull["nodes"]:
                bad = [line for line in out.splitlines() if n["addr"] in line and re.search(r"NG|FAIL|differ|missing", line)]
                n["state"] = "failed" if rc != 0 and (bad or len(nodes) == 1) else "done" if rc == 0 or not bad else "failed"
                n["why"] = bad[0].strip()[:300] if bad else ("" if rc == 0 else last)
            self.pull["state"] = "done" if rc == 0 else "failed"
        log(f"pull on {' '.join(nodes)}: rc {rc}")

    def pull_view(self):
        with self.lock:
            return dict(self.pull, nodes=[dict(x) for x in self.pull["nodes"]])


class Handler(BaseHTTPRequestHandler):
    L: Launcher = None
    server_version = "tf-launcher/1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, status, obj):
        # errors as the engine gives them: {"error": {"message": ...}} (the page reads error.message)
        if status >= 400 and isinstance(obj.get("error"), str):
            obj = dict(obj, error={"message": obj["error"]})
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def route(self, method):
        path = urlparse(self.path).path
        Lr = self.L
        try:
            if method == "GET" and path == "/panel":
                body = open(Lr.page, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(body)
            if method == "GET" and path == "/v1/launch/status":
                return self.send_json(200, Lr.status())
            if method == "GET" and path == "/v1/launch/nodes":
                with Lr.lock:
                    info, t = list(Lr.node_info), Lr.node_t
                nodes = [{k: n.get(k) for k in ("name", "rank", "mem_total", "mem_available", "engine", "other", "gpu_error", "gpu_temp_c", "cpu_temp_c")} for n in info]
                return self.send_json(200, {"age_ms": int((time.time() - t) * 1000) if t else 0, "nodes": nodes})
            if method == "GET" and path == "/v1/launch/usage":
                return self.send_json(200, Lr.usage())
            if method == "GET" and path == "/v1/launch/config":
                return self.send_json(200, Lr.config_view())
            if method == "POST" and path == "/v1/launch/config":
                return self.send_json(*Lr.save_config(self.body()))
            if method == "GET" and path == "/v1/launch/check":
                return self.send_json(200, Lr.check_view())
            if method == "POST" and path == "/v1/launch/check":
                return self.send_json(*Lr.begin_check(self.body()))
            if method == "POST" and path == "/v1/launch/plan":
                Lr.conf_refresh()
                bits, _, o = Lr.bits_of(self.body())
                return self.send_json(200, Lr.plan(bits, **o))
            if method == "POST" and path == "/v1/launch/start":
                err, raw, take = Lr.start_take(self.body())
                if err:
                    return self.send_json(*err)
                bits, name, o = Lr.bits_of(raw)
                return self.send_json(*Lr.begin_start(bits, name, take=take, **o))
            if method == "POST" and path == "/v1/launch/stop":
                return self.send_json(*Lr.begin_stop(self.body()))
            if method == "POST" and path in ("/v1/launch/usage/save", "/v1/launch/usage/discard"):
                self.body()
                return self.send_json(*Lr.pending_decide(path.endswith("/save")))
            if method == "POST" and path == "/v1/launch/allocs/save":
                try:
                    j = json.loads(self.body() or b"{}")
                    engine, with_usage = j.get("engine", False), j.get("with_usage", False)
                    if not isinstance(engine, bool) or not isinstance(with_usage, bool):
                        raise TypeError("engine, with_usage: true or false")
                    bits = None if engine else base64.b64decode(j["bits"], validate=True)  # engine: the engine's bits
                except (ValueError, KeyError, TypeError, AttributeError):
                    raise Fail("BadRequest: JSON, bits, engine or with_usage")
                replace = j.get("replace", False)
                if not isinstance(replace, bool):
                    raise Fail("BadRequest: replace: true or false")
                return self.send_json(*Lr.alloc_save(str(j.get("name", "")), bits, j.get("source", ""), j.get("note", ""), j.get("basis"),
                                                     engine=engine, with_usage=with_usage, replace=replace))
            if method == "GET" and path == "/v1/launch/allocs":
                return self.send_json(200, Lr.allocs_list())
            if method == "GET" and path == "/v1/launch/alloc":
                return self.send_json(*Lr.alloc_one((parse_qs(urlparse(self.path).query).get("name") or [""])[0]))
            if method == "POST" and path == "/v1/launch/allocs/rename":
                return self.send_json(*Lr.alloc_rename(self.body()))
            if method == "POST" and path == "/v1/launch/allocs/delete":
                return self.send_json(*Lr.alloc_delete(self.body()))
            if method == "POST" and path == "/v1/launch/resort":
                return self.send_json(*Lr.resort(self.body()))
            if path in ("/v1/launch/scan", "/v1/launch/ring", "/v1/launch/bw", "/v1/launch/pull") and method == "POST":
                Sv = Lr.servers
                return self.send_json(*{"/v1/launch/scan": lambda: Sv.begin_scan(), "/v1/launch/ring": lambda: Sv.begin_ring(),
                                        "/v1/launch/bw": lambda: Sv.begin_bw(), "/v1/launch/pull": lambda: Sv.begin_pull(self.body())}[path]())
            if method == "POST" and path == "/v1/launch/probe":
                return self.send_json(*Lr.servers.probe_req(self.body()))
            if method == "GET" and path in ("/v1/launch/scan", "/v1/launch/ring", "/v1/launch/links", "/v1/launch/bw", "/v1/launch/pull"):
                Sv = Lr.servers
                return self.send_json(200, {"/v1/launch/scan": Sv.scan_view, "/v1/launch/ring": Sv.ring_view, "/v1/launch/links": Sv.links_view,
                                            "/v1/launch/bw": Sv.bw_view, "/v1/launch/pull": Sv.pull_view}[path]())
            # the node check (the panel's first screen) and where the weights are (the launch settings)
            if path == "/v1/launch/discover" and method in ("GET", "POST"):
                if method == "POST":
                    self.body()
                    return self.send_json(*Lr.servers.begin_discover())
                return self.send_json(200, Lr.servers.disc_view())
            # the inference screen's [ THINKING ]: the engine's change written to glm53.conf too (R-029)
            if method == "POST" and path == "/v1/thinking":
                return self.send_json(*Lr.thinking_post(self.body()))
            if method == "GET" and path == "/v1/launch/weights":
                return self.send_json(200, Lr.weights_get())
            if method == "GET" and path == "/v1/launch/browse":
                return self.send_json(*Lr.browse((parse_qs(urlparse(self.path).query).get("path") or [""])[0]))
            if method == "POST" and path == "/v1/launch/weights/check":
                return self.send_json(*Lr.begin_weights_check(self.body()))
            if method == "POST" and path == "/v1/launch/weights":
                return self.send_json(*Lr.weights_save(self.body()))
            # the engine settings screen (R-024): glm53.conf's keys, written; a stop and a start again with them
            if method == "GET" and path == "/v1/launch/settings":
                return self.send_json(200, Lr.es_view())
            if method == "POST" and path == "/v1/launch/settings":
                return self.send_json(*Lr.es_post(self.body()))
            if method == "POST" and path == "/v1/launch/restart":
                return self.send_json(*Lr.begin_restart(self.body()))
            if path.startswith("/v1/launch/"):
                return self.send_json(404, {"error": "no such launcher path"})
            if method == "GET" and path == "/v1/panel/settings":
                return self.send_json(200, Lr.settings_get())
            if method == "POST" and path == "/v1/panel/settings":
                if int(self.headers.get("Content-Length") or 0) > SETTINGS_MAX:
                    return self.send_json(400, {"error": "the settings are over 64 KiB"})
                return self.send_json(*Lr.settings_post(self.body()))
            if path.startswith("/v1/panel/"):
                return self.send_json(404, {"error": "no such panel path"})
        except Fail as e:
            return self.send_json(400, {"error": str(e)})
        except Exception as e:
            log(f"{method} {path}: {e!r}")
            return self.send_json(500, {"error": repr(e)})
        return self.proxy(method)

    def watch_client(self, upstream, waiting):
        """Until `waiting` is set (the engine's reply began): the client's socket, every second; when it has hung up (a
        read of it gives end-of-file), the engine's connection is shut, which ends the wait in getresponse."""
        sock = self.connection
        while not waiting.wait(1.0):
            try:
                r, _, _ = select.select([sock], [], [], 0)
                if r and sock.recv(1, socket.MSG_PEEK) == b"":
                    log(f"{self.command} {self.path}: the client hung up before the reply: the engine's request closed")
                    if upstream.sock is not None:
                        upstream.sock.shutdown(socket.SHUT_RDWR)
                    return
            except OSError:
                return

    def proxy(self, method):
        """Every other path to the engine (503 while it is not running), streamed back."""
        if not self.L.engine_up:
            return self.send_json(503, {"error": "the engine is not running", "state": self.L.st["state"]})
        body = self.body() if method == "POST" else None
        # a request that is not streamed sends nothing until its reply is whole: a long one (100,000 tokens out is about
        # an hour at TP4) must not time out here
        c = http.client.HTTPConnection(*self.L.engine, timeout=4 * 3600)
        try:
            hdr = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept")}
            c.request(method, self.path, body=body, headers=hdr)
            # while the engine composes a reply that is not streamed, a client that hangs up is passed on: the engine's
            # connection is closed, so the engine cancels the request as it does for its own clients (the
            # generation would otherwise run on to its end for nobody)
            waiting = threading.Event()
            threading.Thread(target=self.watch_client, args=(c, waiting), daemon=True).start()
            try:
                r = c.getresponse()
            finally:
                waiting.set()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() in ("content-type", "cache-control") or k.lower().startswith("x-"):
                    self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError as e:
            try:
                self.send_json(502, {"error": f"the engine: {e}"})
            except OSError:
                pass
        finally:
            c.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--conf", default=os.path.join(os.path.dirname(here), "glm53.conf"))
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--pkg")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    lr = Launcher(a)
    # every address by default: the launcher has no key, so anyone on the network can start, stop and change the
    # engine from the panel (a note, not a refusal; LAUNCHER_HOST=127.0.0.1 keeps it to this node)
    h = lr.listen[0]
    if not (h.startswith("127.") or h in ("localhost", "::1")):
        log(f"note: the panel listens on {h}:{lr.listen[1]} with no key: anyone on the network can start, stop and "
            "change the engine (LAUNCHER_HOST=127.0.0.1 keeps it to this node)")
    # this release's panel is tested on four nodes (TP4); other node counts are refused until they are tested
    # (ATLAS_ANY_WORLD=1 in the environment runs it anyway, for such a test)
    # (a lane not registered yet starts all the same: its panel opens on the node check, which sets the nodes)
    if lr.world not in TESTED_WORLDS and os.environ.get("ATLAS_ANY_WORLD") != "1":
        if lr.registered():
            sys.exit(f"the control panel runs on {' or '.join(map(str, TESTED_WORLDS))} nodes in this release, not {lr.world}: "
                     "start the engine with tf-glm53 engine-up instead")
        log(f"note: {lr.world} nodes in NODES and no {os.path.basename(lr.node_file or 'node file')} yet: the panel opens on the node check")
    why = split_problem(lr.world, lr.dcp, lr.cache)
    if why:
        sys.exit(f"{lr.conf}: {why}")
    Handler.L = lr
    # the port first: a lane that cannot listen stops here, before the threads below log over the reason
    try:
        srv = ThreadingHTTPServer(lr.listen, Handler)
    except OSError as e:
        sys.exit(f"the launcher cannot listen on {lr.listen[0]}:{lr.listen[1]}: {e.strerror or e} (LAUNCHER_PORT sets another port)")
    srv.daemon_threads = True
    lr.probe_all()
    lr.engine_up = lr.health()
    if lr.engine_up:
        lr.st.update(state="running", t0=time.time())
        lr.restore_take()  # what the running start took from saved allocations
    elif any(n["engine"] for n in lr.node_info):
        lr.st.update(state="starting", t0=time.time())
    threading.Thread(target=lr.watch, daemon=True).start()
    threading.Thread(target=lr.poll_nodes, daemon=True).start()
    threading.Thread(target=lr.bank_check, daemon=True).start()  # the banks on the nodes
    shown = lan_name() if lr.listen[0] in ("0.0.0.0", "::") else lr.listen[0]  # every address: the one the LAN reaches
    log(f"launcher for {lr.conf}: http://{shown}:{lr.listen[1]}/panel, engine {lr.engine[0]}:{lr.engine[1]}, "
        f"{lr.world} nodes, state {lr.st['state']}{' (dry run)' if lr.dry else ''}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
