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
  - the nodes: GET /v1/launch/config (NODES in rank order); POST /v1/launch/check {"workers"} runs tf-glm53 deploy and
    check on the head and those workers; POST /v1/launch/config {"workers"} writes NODES once that list passed (the
    head stays; glm53.conf copied aside first; only while stopped)
  - GET / POST /v1/panel/settings: the panel's settings, one JSON document in a file of the lane (TF_PANEL_SETTINGS
    in EXTRA_ENV, else PKG/launch/panel-settings.json), served here whether the engine runs or not; not read, replaced whole
  - every other path goes to the engine (HOST:PORT of glm53.conf), 503 while it is not running: the page has one
    address whether the engine runs or not
  - access: a Bearer key of the lane's API_KEY_FILE, or loopback (403 otherwise); a start or a stop at a time (409)
A start: the pre-flight (every node reachable, its GPU readable by nvidia-smi, no engine or other GPU process on any node, the plan fits; with
LAUNCH_SNAPSHOTS=1 the nodes' state written down, with LAUNCH_DROP_CACHES=1 the page cache dropped), the bits written
as an expert plan (tools/quant/expert_plan.py mix with expert-plan/banks.json), then `tf-glm53 up` with LAUNCH_ENV="TF_GLM53_EXPERT_PLAN=... TF_GLM53_LOAD_PROGRESS=...";
the engine's progress file (glm53-load-progress v1) feeds /v1/launch/status. The ranges (the counts by bits) stay as
the start set them; POST /v1/launch/resort re-sorts the experts between them while the engine runs (the generic kernel,
the swap banks) and saves the allocation, by the decode picks, the prompt picks or both (basis; both weighs decode 0.7, prompt 0.3). A stop: the
usage kept, `tf-glm53 down`, then the run's decode and prompt picks saved into the learning data, dropped, or kept
pending (save_usage true / false / absent; /v1/launch/usage/save
and /discard decide the pending). Kept in PKG/launch/: the learning data (usage-total.json), the pending runs, the last
run, the last measured start (the plan's sizes and the ETA), the plans; allocations in ATLAS_DIR/allocs/user; the
snapshots (LAUNCH_SNAPSHOTS=1) in PKG/snapshots/.
Defaults: CONF = glm53.conf beside this script's directory (PKG = the lane directory), port LAUNCHER_PORT of CONF
(else PORT + 10), host LAUNCHER_HOST (else 127.0.0.1). --pkg: the lane directory on the nodes when this runs elsewhere
(a test from another machine). --dry-run: a start or a stop runs its read-only checks, prints what it would run, and runs
nothing that changes a node.
"""
import argparse, base64, hashlib, json, os, re, select, shlex, shutil, signal, socket, struct, subprocess, sys, threading, time
import http.client
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

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
TESTED_WORLDS = (2, 4, 8)  # the node counts the panel has run on, offered by GET /v1/launch/config (worlds)
# the node counts this engine runs: each expert's 8 stored shares joined 8 / world a rank (2, 4, 8 nodes). Three nodes
# (TP=3: the shares themselves as 3 / 3 / 2 units, heads 22 / 21 / 21) are planned here but need an engine that splits
# the shares so: a start on them is refused until it has one
ENGINE_WORLDS = (2, 4, 8)
# the longest context a start may ask (the engine and the model take 1,048,576 positions); a context is a multiple of
# CONTEXT_STEP positions, at least CONTEXT_MIN
CONTEXT_LIMIT, CONTEXT_STEP, CONTEXT_MIN = 1048576, 16384, 65536
ADDR = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:_-]{0,252}")  # a node's address in NODES (a host name or an IP)
RESERVE = 4 * GiB  # what a node keeps for the OS after the start, by default (RESERVE_GB of CONF, or a plan's reserve_gib);
# the engine's own admission gets the same (TF_GLM53_MEM_RESERVE_GIB)
RESERVE_MIN_GIB, RESERVE_MAX_GIB = 2, 32
ESTIMATE_SLACK = 2 * GiB  # kept beside the reserve when the store takes the rest on an estimate (no measured start)
POOL_MIN_STORE = 512  # the least store with CACHE=pool: the engine needs the context + 8 rows + a block (at most 264)
# the drafters a start may take (tf-glm53 -d): the checkpoint's MTP layer, DFlash2 (DRAFTER of CONF, on every node), none
DRAFTERS = ("mtp", "dflash2", "none")
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


def set_nodes(text, nodes):
    """glm53.conf's text with NODES=(...) set to `nodes` (NODES must be one line of the file)."""
    pat = re.compile(r"^NODES=\([^)\n]*\)", re.M)
    if len(pat.findall(text)) != 1:
        raise Fail("NODES must be one line of glm53.conf, NODES=(...)")
    return pat.sub(lambda m: "NODES=(" + " ".join(nodes) + ")", text)


def read_conf(conf):
    script = r'''. "$1" >/dev/null || exit 1
printf 'NODE=%s\n' "${NODES[@]}"
for v in NAME HOST PORT API_KEY_FILE EXTRA_ENV LAUNCHER_PORT LAUNCHER_HOST KV DCP CONTEXT STORE CACHE RESERVE_GB DRAFTER_KIND DRAFTER ATLAS_DIR ATLAS_GENERIC LAUNCH_DROP_CACHES LAUNCH_SNAPSHOTS RUNTIME IMAGE; do printf '%s=%s\n' "$v" "${!v:-}"; done'''
    out = subprocess.run(["bash", "-c", script, "_", conf], capture_output=True, text=True, check=True).stdout
    c = {"NODES": []}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "NODE":
            c["NODES"].append(v)
        else:
            c[k] = v
    return c


def api_keys(path):
    if not path:
        return set()
    keys = set()
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k = line.split(":", 1)[1].strip() if ":" in line else line
        if k:  # an empty key ("label:") is no key
            keys.add(k)
    return keys


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
        c = read_conf(self.conf)
        self.nodes = c["NODES"]
        self.world = len(self.nodes)
        port = int(c.get("PORT") or 8000)
        host = c.get("HOST") or "127.0.0.1"
        # how tf-glm53 runs the ranks: from the lane directory, or each in a container (RUNTIME=docker: the lane's
        # containers are glm53-p<PORT>-r<rank>, their servers' host process ids the engine's)
        self.runtime = c.get("RUNTIME") or "host"
        self.cname = f"glm53-p{port}-r"
        self.engine = ("127.0.0.1" if host in ("0.0.0.0",) or host.startswith("127.") or host == "localhost" else host, port)
        # the lane's API keys, read again whenever the file changes (keys_now)
        self.key_file = c.get("API_KEY_FILE", "")
        self.key_stamp = None
        self.keys = set()
        self.keys_now()
        self.kv_format = c.get("KV") or "fp8"
        self.dcp = int(c.get("DCP") or (3 if self.world == 3 else 2))  # tf-glm53's default
        # the positions CONF asks for: a request's context and the conversation store beside it; a start in a format
        # that cannot hold them all gets a shorter context (kv_choice)
        self.context = int(c.get("CONTEXT") or 819200)
        self.store = int(c.get("STORE") or 819200)
        self.cache = c.get("CACHE") or "pool"  # tf-glm53's default
        # what a node keeps for the OS after a start (a plan may ask another: reserve_gib)
        self.reserve = int(float(c.get("RESERVE_GB") or RESERVE / GiB) * GiB)
        self.kv_positions_now = self.context + self.store
        # the drafter CONF names (a start may take another: drafter) and DFlash2's model, checked on every node with
        # the banks (drafter_state)
        self.drafter_conf = c.get("DRAFTER_KIND") or "mtp"
        self.drafter_path = c.get("DRAFTER") or ""
        self.drafter_state = {}
        # the nodes' check (POST /v1/launch/check): the last result, one at a time
        self.check = {"state": "idle", "started_unix": 0, "ended_unix": 0, "nodes": [], "ok": False, "why": "", "workers": []}
        self.check_lock = threading.Lock()
        env = dict(t.split("=", 1) for t in shlex.split(c.get("EXTRA_ENV", "")) if "=" in t)
        self.page = env.get("TF_PANEL_HTML", os.path.join(self.pkg_local, "panel.html"))
        self.sets = env.get("TF_GLM53_EXPERT_SETS", os.path.join(self.pkg_local, "expert-sets"))
        self.listen = (a.host or c.get("LAUNCHER_HOST") or "127.0.0.1", a.port or int(c.get("LAUNCHER_PORT") or port + 10))
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
        # what the start took from saved allocations (no_take: alloc, bank_ids, usage_from, base)
        self.st = dict({"state": "stopped", "phase": "", "t0": 0.0, "bits": None, "set": "", "error": "", "seen": {}}, **no_take())
        self.job = None  # the start or stop thread
        self.proc = None  # tf-glm53 up / down
        self.cancel = False
        self.engine_up = False
        self.node_info = []
        self.node_t = 0.0
        self.names = list(self.nodes)
        self.engine_bits = None
        self.start_before = {}  # each node's MemAvailable before the current start
        self.usage_lock = threading.RLock()  # keep_usage holds it around end_run
        self.completions = None  # the engine's completion_tokens_total at the last /health
        # Expert Atlas's own directory, apart from the lane's: saved allocations in allocs/{builtin,user}
        self.atlas = c.get("ATLAS_DIR") or os.path.join(self.pkg_local, "atlas")
        self.alloc_lock = threading.Lock()  # one save, rename or delete of an allocation at a time
        # opt-in steps of a start or a stop: the page cache dropped (sudo), the nodes' state written down (snapshot)
        self.drop_caches = (c.get("LAUNCH_DROP_CACHES") or "0") == "1"
        self.snapshots = (c.get("LAUNCH_SNAPSHOTS") or "0") == "1"
        # every layer on the generic grouped kernel, so a re-sort may move any expert
        self.generic = (c.get("ATLAS_GENERIC") or "1") == "1"

    def ssh(self, node, cmd, timeout=20):
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
        with ThreadPoolExecutor(self.world) as ex:
            info = list(ex.map(self.probe, range(self.world)))
        with self.lock:
            self.node_info, self.node_t = info, time.time()
        return info

    def on_all(self, cmd, timeout=120):
        with ThreadPoolExecutor(self.world) as ex:
            return list(ex.map(lambda n: self.ssh(n, cmd, timeout), self.nodes))

    def keys_now(self):
        """The lane's API keys as the file holds them now: read again when its identity, size or time changes, as the
        engine reads its own (a key taken out loses the launcher's control, a new one gains it at once).
        A file that cannot be read gives no key: only loopback is let in until it reads again."""
        if not self.key_file:
            return self.keys
        try:
            s = os.stat(self.key_file)
            stamp = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if stamp != self.key_stamp:
                keys = api_keys(self.key_file)
                if self.key_stamp is not None:
                    log(f"API key file read again: {len(keys)} keys")
                self.keys, self.key_stamp = keys, stamp
        except OSError as e:
            if self.keys or self.key_stamp is not None:
                log(f"API key file {self.key_file} cannot be read ({e}): no key accepted until it can")
            self.keys, self.key_stamp = set(), None
        return self.keys

    def engine_headers(self):
        """The launcher's own calls to the engine carry a key of the lane's file when it has keys (the engine asks for
        one from loopback too)."""
        keys = self.keys_now()
        return {"Authorization": "Bearer " + next(iter(sorted(keys)))} if keys else {}

    def engine_get(self, path, timeout=3):
        c = http.client.HTTPConnection(*self.engine, timeout=timeout)
        try:
            c.request("GET", path, headers=self.engine_headers())
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def health(self):
        """Whether the engine answers /health; its completion_tokens_total is kept (self.completions) so the usage is
        read again soon after the engine decodes."""
        try:
            st, body = self.engine_get("/health", 2)
            if st != 200:
                return False
            try:
                self.completions = json.loads(body).get("completion_tokens_total", self.completions)
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
            p = os.path.join(self.dir, "pending.json")
            os.replace(p, p + time.strftime(".%Y%m%d-%H%M%S.") + ("saved" if save else "discarded"))
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
        last_usage, down, kept_at = time.time(), 0, None
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
            # the run's usage read again 2 s after the engine decoded anything (else every 5 minutes): a stop
            # elsewhere or a crash, which leaves nothing to read, loses at most those seconds (10 s lost most of a short
            # reply)
            moved = self.completions is not None and self.completions != kept_at
            if up and (time.time() - last_usage > 300 or self.engine_bits is None or (moved and time.time() - last_usage >= 2)):
                c = self.completions
                self.keep_usage()  # also the bits of an engine the launcher did not start
                last_usage, kept_at = time.time(), c
            time.sleep(1)

    def poll_nodes(self):
        while True:
            try:
                self.probe_all()
            except Exception as e:  # keep polling
                log(f"nodes: {e}")
            time.sleep(5)

    def status(self):
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
                # the KV the start took: this launcher's start, else the last measured one (a launcher restarted while
                # the engine runs), else CONF's
                "kv": s.get("kv") or (self.read_json("last-start.json") or {}).get("kv") or {"format": self.kv_format, "context": self.context, "store": self.store, "dcp": self.dcp},
                # the widths whose banks every node has (the launch screen offers only those)
                "banks": self.bank_view(),
                # the drafters a start may take, and the one this start or the last measured one took
                "drafter": dict(self.drafter_view(), running=s.get("drafter") or (self.read_json("last-start.json") or {}).get("drafter", self.drafter_conf))}

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

    def kv_choice(self, fmt, room, drafter=None, context=None, store=None):
        """The context and store a start in `fmt` takes with `room` bytes for the KV caches on the tightest node: the
        asked context (CONF's when None) when it fits with the store, else the context cut to what fits (a multiple of
        16,384, the store kept); None when not even 65,536 tokens of context fit. The store: CONF's (None), the one
        asked, or "max": everything the context leaves (a multiple of 16,384)."""
        want = context or self.context
        keep = self.store_floor() if store == "max" else (self.store if store is None else store)
        ctx = self.max_context(fmt, room, drafter, keep)
        if not ctx:
            return None
        ctx = min(ctx, want)
        if store == "max":
            return ctx, self.store_max(room // self.position_bytes(drafter)[fmt], ctx)
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

    def max_context(self, fmt, room, drafter=None, store=None):
        """The longest context that fits `room` with `store` (CONF's when None) in `fmt` (a multiple of 16,384, at most
        the engine's limit; 0 below 65,536)."""
        most = room // self.position_bytes(drafter)[fmt]
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

    def plan(self, bits, nodes=None, kv=None, drafter=None, context=None, store=None, reserve_gib=None):
        """The plan of a start with `bits`, the KV format `kv` and the drafter `drafter` (CONF's when None): each node's
        memory with the KV caches the format's context and store take, the verdict, and both formats' positions (the
        launch screen picks one). `context`: the context asked (CONF's when None), cut when it does not fit; `store`: the
        store asked (CONF's when None, "max": the rest); `reserve_gib`: what a node keeps for the OS (CONF's when None)."""
        fmt = kv or self.kv_format
        drafter = drafter or self.drafter_conf
        default = defaults(self.world, drafter)
        reserve = int(reserve_gib * GiB) if reserve_gib is not None else self.reserve
        nodes = nodes or self.node_info or self.probe_all()
        last = self.last_start(drafter)
        # the store taking the rest on an estimate keeps ESTIMATE_SLACK more (the sizes not yet measured)
        slack = ESTIMATE_SLACK if store == "max" and (last is None or last.get("world") != self.world) else 0
        estimate = last is None or last.get("world") != self.world
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
            # what the KV caches could take with these bits: the node less everything but the KV, less the reserve
            room = max(0, avail - ex - sizes["other_weights"] - sizes["work"] - reserve - slack)
            out.append({"name": n["name"], "experts": ex, "other_weights": sizes["other_weights"], "work": sizes["work"],
                        "available": avail, "kv_room": room})
        room = min((n["kv_room"] for n in out), default=0)
        pb = self.position_bytes(drafter)
        # each format: the positions the start would allocate (CONF's, or a shorter context) and their bytes a node
        formats = {}
        for f in pb:
            ch = self.kv_choice(f, room, drafter, context, store)
            ctx, st = ch if ch else (0, self.store if store in (None, "max") else store)
            keep = 0 if store == "max" else (self.store if store is None else store)
            formats[f] = {"context": ctx, "store": st, "positions": ctx + st if ch else 0, "bytes": (ctx + st) * pb[f] if ch else 0,
                          "max_positions": room // pb[f], "max_context": self.max_context(f, room, drafter, keep),
                          "max_store": self.store_max(room // pb[f], ctx) if ch else 0,
                          "cut": bool(ch) and ctx < (context or self.context), "fits": ch is not None}
        kvb = formats[fmt]["bytes"]
        for n in out:
            n["kv"] = kvb
            n["total"] = n["experts"] + n["other_weights"] + kvb + n["work"]
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
        miss = self.missing_bits(bits)
        if miss:
            worst = "no"
        dv = self.drafter_view()
        if drafter not in dv["available"]:
            worst = "no"
        return {"verdict": worst, "reserve": reserve, "conf_reserve": self.reserve, "reserve_gib": reserve / GiB,
                "estimate": bool(estimate), "nodes": out, "kv": kv,
                "missing_bits": {str(w): why for w, why in miss.items()}, "drafter": dict(dv, kind=drafter)}

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
            b = base64.b64decode(j["bits"], validate=True)
        except (KeyError, ValueError, TypeError):
            raise Fail("bits: base64 expected")
        if len(b) != L * E or any(v < 2 or v > 5 for v in b):
            raise Fail("bits: length or value")
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
        return b, name, {"kv": kv, "drafter": drafter, "context": context, "store": store, "reserve_gib": reserve}

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

    def begin_start(self, bits, name, kv=None, drafter=None, context=None, store=None, reserve_gib=None, take=None):
        # two starts at once: the second one 409, not a second worker
        if not self.op_lock.acquire(blocking=False):
            return 409, {"error": "a start is being admitted"}
        try:
            return self.begin_start_locked(bits, name, kv, drafter, context, store, reserve_gib, take or no_take())
        finally:
            self.op_lock.release()

    def begin_start_locked(self, bits, name, kv=None, drafter=None, context=None, store=None, reserve_gib=None, take=None):
        with self.lock:
            if self.st["state"] not in ("stopped", "failed") or (self.job and self.job.is_alive()):
                return 409, {"error": f"not stopped ({self.st['state']})"}
        if self.world not in ENGINE_WORLDS:
            return 412, {"error": f"this engine does not run on {self.world} nodes yet (TP={self.world} needs the three-way engine): "
                                  "plans only, a start is refused"}
        kv = kv or self.kv_format
        drafter = drafter or self.drafter_conf
        # the banks (and DFlash2's model) checked again now: a width without data refused here, not after the ranks start
        miss = self.missing_bits(bits, force=True)
        if miss:
            return 412, {"error": "no data for " + ", ".join(f"{w} bit ({why})" for w, why in miss.items()), "missing_bits": {str(w): why for w, why in miss.items()}}
        # every node read now, not the last poll: right after a stop the poll may have caught the memory still being freed
        dv = self.drafter_view()
        if drafter not in dv["available"]:
            return 412, {"error": f"drafter {drafter}: " + dv["why"].get(drafter, "not available"), "drafter": dv}
        p = self.plan(bits, nodes=self.probe_all(), kv=kv, drafter=drafter, context=context, store=store, reserve_gib=reserve_gib)
        if p["verdict"] == "no":
            return 412, {"error": "the plan does not fit", "plan": p}
        take = take or no_take()
        with self.lock:
            self.cancel = False
            self.st.update(state="starting", phase="checking", t0=time.time(), bits=bits, set=name, error="",
                           seen={"checking": time.time()}, **take)
            self.job = threading.Thread(target=self.run_start, args=(bits, name, kv, drafter, context, store, reserve_gib, take), daemon=True)
            self.job.start()
        return 200, {"state": "starting"}

    def set_phase(self, phase):
        with self.lock:
            self.st["phase"] = phase
            self.st["seen"].setdefault(phase, time.time())

    def run_start(self, bits, name, kv, drafter=None, context=None, store=None, reserve_gib=None, take=None):
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            self.preflight(ts)
            before = {n["name"]: n["mem_available"] for n in self.probe_all()}
            with self.lock:
                self.start_before = dict(before)
            drafter = drafter or self.drafter_conf
            p = self.plan(bits, kv=kv, drafter=drafter, context=context, store=store, reserve_gib=reserve_gib)
            if p["verdict"] == "no":
                worst = min(p["nodes"], key=lambda n: n["margin"])
                raise Fail(f"the plan does not fit: {worst['name']} margin {worst['margin'] / GiB:.1f} GiB")
            # the KV format the launch screen chose, and the context and store it takes (tf-glm53 reads LAUNCH_KV,
            # LAUNCH_CONTEXT, LAUNCH_STORE over CONF's KV, CONTEXT, STORE)
            # and what a node must have available before it (tf-glm53 check reads LAUNCH_NEED_GB over its own estimate
            # from CONF's STORE): the plan's sizes and the reserve, on the node that needs most
            need = max(n["total"] for n in p["nodes"]) + p["reserve"]
            kvs = {"LAUNCH_KV": kv, "LAUNCH_CONTEXT": str(p["kv"]["context"]), "LAUNCH_STORE": str(p["kv"]["store"]),
                   "LAUNCH_NEED_GB": str(int(need // GiB))}
            with self.lock:
                self.st["kv"] = {"format": kv, "context": p["kv"]["context"], "store": p["kv"]["store"], "dcp": self.dcp,
                                 "reserve_gib": p["reserve"] / GiB}
                self.st["drafter"] = drafter
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
                raise Fail((why + " / " if why else "") + (tail or f"tf-glm53 up exited {rc}"))
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
        """GET /v1/launch/config: NODES in rank order, each node's host name once probed ("" before), the world, and
        when glm53.conf was saved."""
        try:
            saved = int(os.path.getmtime(self.conf))
        except OSError:
            saved = 0
        with self.lock:
            nodes, names = list(self.nodes), list(self.names)
        return {"nodes": nodes, "names": [n if n != a else "" for n, a in zip(names, nodes)], "world": self.world,
                "worlds": list(TESTED_WORLDS), "saved_unix": saved}

    def workers_of(self, body, required):
        """The workers a check or a save names: w - 1 distinct addresses for a node count w the panel offers (worlds; a
        save of another count changes the world), the head not among them (CONF's when the body names none and none is
        required)."""
        try:
            j = json.loads(body or b"{}")
        except ValueError:
            raise Fail("bad JSON")
        w = j.get("workers") if isinstance(j, dict) else None
        if w is None and not required:
            return list(self.nodes[1:])
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
        if self.nodes[0] in w:
            raise Fail("workers: the head is not a worker")
        return w

    def stopped(self):
        with self.lock:
            return self.st["state"] in ("stopped", "failed") and not (self.job and self.job.is_alive()) and not self.engine_up

    def check_view(self):
        with self.check_lock:
            return {k: v for k, v in self.check.items()}

    def begin_check(self, body):
        """POST /v1/launch/check: tf-glm53 deploy, then tf-glm53 check, on the head and these workers, in the
        background (the package goes to a new worker first, or its files could not pass)."""
        workers = self.workers_of(body, required=False)
        if not self.stopped():
            return 409, {"error": "the engine is not stopped"}
        with self.check_lock:
            if self.check["state"] == "checking":
                return 409, {"error": "a check is running"}
            self.check = {"state": "checking", "started_unix": int(time.time()), "ended_unix": 0, "nodes": [], "ok": False,
                          "why": "", "workers": workers}
        threading.Thread(target=self.run_check, args=(workers,), daemon=True).start()
        return 202, {"state": "checking"}

    def run_check(self, workers):
        nodes = [self.nodes[0]] + workers
        res = [{"rank": r, "addr": a, "name": "", "ok": False, "why": "", "detail": ""} for r, a in enumerate(nodes)]
        why = []
        tmp = os.path.join(self.dir, "check-" + time.strftime("%Y%m%d-%H%M%S") + ".conf")
        try:
            open(tmp, "w").write(set_nodes(open(self.conf).read(), nodes))
            tf = ["bash", os.path.join(self.pkg_local, "tf-glm53"), "-c", tmp]
            d = subprocess.run(tf + ["deploy"], capture_output=True, text=True, timeout=1800)
            c = subprocess.run(tf + ["check"], capture_output=True, text=True, timeout=600)
            for line in d.stdout.splitlines():  # "  FAIL ADDR": the package could not be copied there
                m = re.match(r"\s+FAIL (\S+)$", line)
                if m and m.group(1) in nodes:
                    res[nodes.index(m.group(1))]["why"] = "the package could not be copied (tf-glm53 deploy)"
            if "SHA256SUMS does not match here" in d.stdout:
                why.append("this directory does not match its SHA256SUMS")
            for line in c.stdout.splitlines():
                m = re.match(r"\s+rank (\d+) (\S+): (ok|NG)(?::|\s)\s*(.*)$", line)
                if m and int(m.group(1)) < len(res):
                    r = res[int(m.group(1))]
                    if m.group(3) == "ok":
                        r["ok"], r["detail"] = not r["why"], m.group(4).strip().strip("()")
                    else:
                        r["why"] = ", ".join(x for x in (r["why"], m.group(4).strip()) if x)
                elif line.strip() and not re.match(r"\s*(drafter:|check: ok)", line) and not m:
                    if c.returncode != 0 and not line.startswith("check:"):
                        why.append(line.strip())
            with ThreadPoolExecutor(len(nodes)) as ex:
                names = list(ex.map(lambda a: self.ssh(a, "hostname", 15)[1].strip(), nodes))
            for r, n in zip(res, names):
                r["name"] = n
                if not r["ok"] and not r["why"]:
                    r["why"] = "not reached by tf-glm53 check"
            ok = c.returncode == 0 and d.returncode == 0 and all(r["ok"] for r in res)
        except (OSError, subprocess.TimeoutExpired, Fail) as e:
            why.append(str(e))
            ok = False
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        with self.check_lock:
            self.check.update(state="done", ended_unix=int(time.time()), nodes=res, ok=ok, why="; ".join(why[:5]))
        log(f"check of {' '.join(nodes)}: {'ok' if ok else 'NG'}" + ("" if ok else ": " + "; ".join(f"{r['addr']}: {r['why']}" for r in res if r["why"])))

    def save_config(self, body):
        """POST /v1/launch/config: NODES written as the head and these workers, once a check passed on exactly this
        list; glm53.conf copied aside first, written whole (a temporary file, then a rename), then read again."""
        workers = self.workers_of(body, required=True)
        if not self.stopped():
            return 409, {"error": "the engine is not stopped"}
        ch = self.check_view()
        if ch["state"] != "done" or not ch["ok"] or ch.get("workers") != workers:
            return 412, {"error": "check this list first (POST /v1/launch/check), then save it"}
        nodes = [self.nodes[0]] + workers
        text = set_nodes(open(self.conf).read(), nodes)
        keep = self.conf + time.strftime(".bak-%Y%m%d-%H%M%S")
        shutil.copy2(self.conf, keep)
        tmp = f"{self.conf}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, self.conf)
        self.reload_nodes()
        log(f"NODES saved: {' '.join(nodes)} (the previous glm53.conf in {os.path.basename(keep)})")
        return 200, self.config_view()

    def reload_nodes(self):
        """glm53.conf's NODES read again: probes, the banks' and the drafter's checks and plans use the new nodes."""
        c = read_conf(self.conf)
        with self.lock:
            self.nodes = c["NODES"]
            self.world = len(self.nodes)
            self.names = list(self.nodes)
            self.node_info, self.node_t = [], 0.0
        with self.bank_lock:
            self.bank_state, self.bank_t, self.drafter_state = {}, 0.0, {}
        threading.Thread(target=self.bank_check, kwargs={"force": True}, daemon=True).start()
        threading.Thread(target=self.probe_all, daemon=True).start()

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
        arg = base64.b64encode(json.dumps({"banks": [{"bits": b["bits"], "dir": b["dir"], "manifest": manifest_of(b)} for b in banks],
                                           "drafter": self.drafter_path}).encode()).decode()
        code = base64.b64encode(BANK_PROBE.encode()).decode()
        res = self.on_all(f"python3 -c \"$(echo {code} | base64 -d)\" {arg}", 60)
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

    def bank_view(self):
        """The launch screen's view of the banks (status, plan): the widths and whether each can be used; checked again
        in the background once BANK_TTL has passed, so a status never waits on ssh."""
        with self.bank_lock:
            state, t, busy = self.bank_state, self.bank_t, self.bank_busy
        if (time.time() - t >= BANK_TTL or self.banks_changed()) and not busy:
            threading.Thread(target=self.bank_check, daemon=True).start()
        return {"checked_unix": int(t), "checking": busy, "bits": {str(w): {"id": s["id"], "ok": s["ok"],
                "why": "; ".join(f"{n}: {v}" for n, v in s["nodes"].items() if v != "ok")} for w, s in sorted(state.items())},
                "available": [w for w, s in sorted(state.items()) if s["ok"]]}

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
        """One bank a width from expert-plan/banks.json (the first listed)."""
        seen, out = set(), []
        for b in json.load(open(self.banks))["banks"]:
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
        logs = os.path.join(self.pkg, "logs")
        seen, out = set(), []
        for i, (rc, line) in enumerate(self.on_all(f"grep -m1 -h '^error: glm53' {logs}/rank*.log 2>/dev/null", 20)):
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                out.append(f"{self.names[i]}: {line}")
        return " / ".join(out[:3])

    def run_tf(self, what, env, ts, extra=None, drafter=None):
        """tf-glm53 up / down (up with -d DRAFTER), its output to PKG/logs/launch-<what>-<ts>.out; the exit code and
        the last lines."""
        cmd = ["bash", os.path.join(self.pkg_local, "tf-glm53"), "-c", self.conf] + (["-d", drafter] if drafter else []) + [what]
        if self.dry:
            log(f"dry run: LAUNCH_ENV={shlex.quote(env)} {' '.join(f'{k}={v}' for k, v in (extra or {}).items())} {' '.join(cmd)}")
            raise Fail("dry run: nothing was started" if what == "up" else "dry run: nothing was stopped")
        os.makedirs(os.path.join(self.pkg_local, "logs"), exist_ok=True)
        path = os.path.join(self.pkg_local, "logs", f"launch-{what}-{ts}.out")
        lines = []
        with open(path, "w") as out:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    env=dict(os.environ, LAUNCH_ENV=env, **(extra or {})))
            with self.lock:
                self.proc = proc

            def read():
                for line in proc.stdout:
                    out.write(line)
                    out.flush()
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
        keep = [x for x in lines if x and not re.fullmatch(r"\s*\d+s", x)]
        return rc, " / ".join(keep[-4:])

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
        nodes = {}
        for name, a0 in before.items():
            took = a0 - after.get(name, a0)
            nodes[name] = {"avail_before": a0, "avail_after": after.get(name, 0), "other_weights": other, "kv": kv,
                           "work": max(0, took - ex - other - kv)}
        with self.lock:
            seen = dict(self.st["seen"])
        seen["done"] = time.time()
        phases = {k: seen[PHASES[i + 1] if i + 1 < len(PHASES) else "done"] - seen[k]
                  for i, k in enumerate(PHASES) if k in seen and (PHASES[i + 1] if i + 1 < len(PHASES) else "done") in seen}
        with self.lock:
            kvu = self.st.get("kv")
        doc = {"unix": int(time.time()), "world": self.world, "nodes": nodes, "phases": phases, "kv": kvu,
               "drafter": drafter or self.drafter_conf}
        self.write_json("last-start.json", doc)
        self.write_json(f"last-start-{doc['drafter']}.json", doc)  # the plans of later starts with this drafter
        self.keep_usage()

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
                    self.proc.send_signal(signal.SIGTERM)  # tf-glm53 up stops every rank on TERM
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
        if not j.get("apply"):
            return 200, out
        if miss:
            return 412, dict(out, error="no data for " + ", ".join(f"{w} bit ({why})" for w, why in miss.items()))
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
            c.request("POST", "/v1/experts/swap", body=json.dumps({"bits": out["target_bits"]}), headers={"Content-Type": "application/json", **self.engine_headers()})
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

    def snapshot(self, label):
        """LAUNCH_SNAPSHOTS=1: each node's GPU, memory, containers and log tail before and after a start or a stop, in
        PKG/snapshots/<time>-<what>/ (off by default)."""
        if not self.snapshots:
            return
        d = os.path.join(self.pkg_local, "snapshots", label)
        os.makedirs(d, exist_ok=True)
        cmd = ("echo '## nvidia-smi'; nvidia-smi; echo '## free -h'; free -h; echo '## docker ps'; docker ps 2>&1; "
               f"echo '## log tail'; tail -n 5 {self.pkg}/logs/rank*.log 2>/dev/null")
        for name, (rc, out) in zip(self.names, self.on_all(cmd, 60)):
            open(os.path.join(d, f"{name}.txt"), "w").write(out)
        h = "200" if self.health() else "down"
        open(os.path.join(d, "health.txt"), "w").write(f"{self.engine[0]}:{self.engine[1]}/health {h}\n")


class Handler(BaseHTTPRequestHandler):
    L: Launcher = None
    server_version = "tf-launcher/1"

    def log_message(self, fmt, *args):
        pass

    def allowed(self):
        if self.client_address[0] in ("127.0.0.1", "::1"):
            return True
        auth = self.headers.get("Authorization", "")
        return auth.startswith("Bearer ") and auth[7:].strip() in self.L.keys_now()

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
        # the page is static and holds no data: served without a key, so a browser can open /panel#key=... and the page
        # sends the key with every API call; everything else needs the key or loopback
        if not (method == "GET" and path == "/panel") and not self.allowed():
            return self.send_json(403, {"error": "a key or loopback"})
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
            hdr = {k: v for k, v in self.headers.items() if k.lower() in ("authorization", "content-type", "accept")}
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
    # this release's panel is tested on four nodes (TP4); other node counts are refused until they are tested
    # (ATLAS_ANY_WORLD=1 in the environment runs it anyway, for such a test)
    if lr.world not in TESTED_WORLDS and os.environ.get("ATLAS_ANY_WORLD") != "1":
        sys.exit(f"the control panel runs on {' or '.join(map(str, TESTED_WORLDS))} nodes in this release, not {lr.world}: "
                 "start the engine with tf-glm53 up instead")
    why = split_problem(lr.world, lr.dcp, lr.cache)
    if why:
        sys.exit(f"{lr.conf}: {why}")
    Handler.L = lr
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
    srv = ThreadingHTTPServer(lr.listen, Handler)
    srv.daemon_threads = True
    log(f"launcher for {lr.conf}: http://{lr.listen[0]}:{lr.listen[1]}/panel, engine {lr.engine[0]}:{lr.engine[1]}, "
        f"{lr.world} nodes, state {lr.st['state']}{' (dry run)' if lr.dry else ''}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
