#!/usr/bin/env python3
"""quickstart: from a fresh install to a running engine, on this node (the head), through this lane's launcher.

Run by `tf-glm53 quickstart`, which `docker run ... IMAGE up` calls over ssh on the host; also by hand. It prints one
numbered line a step. On anything it cannot do itself it prints STOP (what is wrong), FIX (the exact thing to do) and
THEN (what to run next), says where the dashboard and the guide are, and leaves the launcher running. Exit 0: the
engine runs; 2: a fix is needed.

The checkpoint and the 3-bit bank are always needed; every expert loads at 3 bits unless BITS or ALLOC says otherwise.
Options, each --set NAME=VALUE (`docker run -e NAME=VALUE`; QUICKSTART.md lists them):
  WEIGHTS, BANK_2, BANK_3, BANK_4, BANK_5, DRAFTER   where the files are, the same path on every node
  BITS=2..5, ALLOC=NAME, NODES="A B C"               every expert at that width; a saved allocation; a switch's workers
  KV_SIZE=N or max                                   the KV size for this start (none given: max, all the memory left)
  any other key of the engine settings (CONFIG.md)   saved in glm53.conf, as the settings screen saves it
"""
import argparse, base64, difflib, json, os, re, socket, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

LAYERS, EXPERTS = 75, 256
DEFAULT_BITS = 3
GUIDE = ("QUICKSTART.md", "AGENT-API.md", "AI-GUIDE.md")
CHECKPOINT_REPO = {4: "fiore-lab/GLM-5.3-EXL3-4x-DGX-Sparks", 8: "fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks"}
BANK_OPTIONS = {f"BANK_{w}": w for w in (2, 3, 4, 5)}
# this script's own options; the rest are the settings' keys
OWN_OPTIONS = ("WEIGHTS", "DRAFTER", "BITS", "ALLOC", "NODES", *BANK_OPTIONS)
ACCESS_WHY = {"hostkey": "its host key is not known here", "nokey": "it refuses this node's ssh key", "noanswer": "no answer on ssh", "noaddr": "its LAN address is not known here"}
# what the node check decides and writes (glm53.nodes.conf); no option sets them
NODE_CHECK_KEYS = ("TOPOLOGY", "SOCKET_IFNAME", "IB_HCA")


def say(s=""):
    print(s, flush=True)


class Stop(Exception):
    """what is wrong, the exact fix, what to run next"""
    def __init__(self, what, fix, then=None):
        super().__init__(what)
        self.what, self.fix, self.then = what, fix, then


def lan_address():
    """The address other machines on the LAN reach this node at (the default route's source)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 9))
        a = s.getsockname()[0]
        s.close()
        return a
    except OSError:
        o = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        return o[0] if o else socket.gethostname()


class Quick:
    def __init__(self, a):
        self.a = a
        self.pkg = a.pkg
        self.lan = lan_address()
        self.lp, self.ep = a.launcher_port, a.api_port
        self.again = a.again or f"cd {a.pkg} && ./tf-glm53 quickstart"
        self.world = 0
        self.listening = False  # this lane's launcher answers on its port
        self.notes = []
        self.opts = {}
        for kv in a.set:
            k, _, v = kv.partition("=")
            self.opts[k] = v

    def call(self, method, path, body=None, timeout=60):
        """The launcher: (HTTP code, the answer's JSON or {"raw": text})."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.lp}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                code, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, raw = e.code, e.read()
        try:
            return code, json.loads(raw)
        except ValueError:
            return code, {"raw": raw[:400].decode(errors="replace")}

    @staticmethod
    def why(r):
        e = r.get("error", r)
        return e.get("message", e) if isinstance(e, dict) else e

    def log_fix(self, which="launcher"):
        return f"read the log: cd {self.pkg} && ./tf-glm53 logs {which}"

    def wait(self, path, done, what, every=2, limit=900):
        t0 = time.time()
        while time.time() - t0 < limit:
            d = self.call("GET", path)[1]
            if done(d):
                return d
            time.sleep(every)
        raise Stop(f"{what}: no answer after {limit} s", self.log_fix())

    def tf(self, *args):
        return subprocess.run([os.path.join(self.pkg, "tf-glm53"), "-c", self.a.conf, *args], capture_output=True, text=True)

    # step 1 -------------------------------------------------------------------------------------------------------------
    def launcher(self):
        r = self.tf("launcher", "start")
        if r.returncode != 0:
            out = (r.stdout + r.stderr).strip()
            if "cannot listen on" in out:
                raise Stop(f"port {self.lp} is in use: this lane's launcher cannot listen there",
                           "stop what holds it (another lane's launcher: ./tf-glm53 launcher stop, in its directory), or add -e PORT=N to the docker run line (the API at N, the launcher at N + 10)", self.again)
            raise Stop("the launcher did not start: " + out[-400:], self.log_fix())
        for _ in range(30):
            try:
                code, c = self.call("GET", "/v1/launch/config", timeout=10)
                if code == 200:
                    # a port another lane holds: that lane's launcher answers, and this one could not listen
                    if os.path.realpath(c.get("conf_path", "")) != os.path.realpath(self.a.conf):
                        raise Stop(f"port {self.lp} answers another lane's launcher ({c.get('conf_path') or 'an older build'})",
                                   "stop that launcher (in its directory: ./tf-glm53 launcher stop), or add -e PORT=N to the docker run line (the API at N, the launcher at N + 10)", self.again)
                    self.listening = True
                    say(f"[1/6] launcher: running, on this node, port {self.lp}")
                    return
            except OSError:
                pass
            time.sleep(2)
        raise Stop(f"the launcher does not answer on port {self.lp}", self.log_fix())

    def check_options(self):
        """Every option known, before anything changes: a typo stops here."""
        v = self.call("GET", "/v1/launch/settings")[1]
        self.settings = {k["key"]: k for k in v.get("keys", [])}
        known = set(OWN_OPTIONS) | {k for k, x in self.settings.items() if x.get("when") != "readonly"}
        bad = []
        for k in self.opts:
            if k in NODE_CHECK_KEYS:
                bad.append(f"{k}: the node check decides it from the cabling; no option sets it")
            elif k not in known:
                near = difflib.get_close_matches(k, sorted(known), n=2, cutoff=0.6)
                bad.append(f"{k}: not an option" + (f" (did you mean {' or '.join(near)}?)" if near else ""))
        if self.opts.get("BITS", "3") not in ("2", "3", "4", "5"):
            bad.append(f"BITS={self.opts['BITS']}: a width, 2 to 5")
        if "BITS" in self.opts and "ALLOC" in self.opts:
            bad.append("BITS and ALLOC: one or the other (an allocation holds its own widths)")
        if bad:
            raise Stop("options this start cannot take:\n" + "\n".join("  " + x for x in bad), f"every option: {self.pkg}/{GUIDE[0]}, section on options", self.again)

    # step 2 -------------------------------------------------------------------------------------------------------------
    def nodes(self):
        c = self.call("GET", "/v1/launch/config")[1]
        want = (self.opts.get("NODES") or "").replace(",", " ").split()
        registered = c.get("nodes") or []
        # the nodes asked for are the workers, the head among them or not: fewer than are registered registers them again
        if c.get("registered") and (not want or set(want) - set(registered[:1]) == set(registered[1:])):
            self.world = c["world"]
            say(f"[2/6] nodes: registered already: {c['topology']}, {' '.join(n or a for n, a in zip(c.get('names') or c['nodes'], c['nodes']))}")
            # stopped: this directory (an update changes it) and the image on every node again; running: nothing to touch
            if self.call("GET", "/v1/launch/status")[1].get("state") in ("stopped", "failed"):
                self.ready_nodes(c["nodes"][0], c["nodes"][1:], registered=True)
            return
        st = self.call("GET", "/v1/launch/status")[1]
        if st.get("state") != "stopped":
            raise Stop(f"the nodes cannot change while the engine is {st.get('state')}", f"stop the engine first: {self.pkg}/{GUIDE[1]} section 7 (saves the usage first)", self.again)
        say("[2/6] nodes: looking at this node's ConnectX-7 links (read only)")
        code, d = self.call("POST", "/v1/launch/discover", {})
        if code not in (200, 202):
            raise Stop(f"discover refused ({code}): {self.why(d)}", self.log_fix())
        d = self.wait("/v1/launch/discover", lambda x: x.get("state") == "done", "discover")
        kind = d.get("kind")
        say(f"      cabling: {kind}" + (f" ({d['why']})" if d.get("why") else ""))
        if kind == "none":
            raise Stop("no other node answers over ConnectX-7", f"check the cables and the link addresses on this node (ip -br addr; rdma link), and that this node can ssh to itself and to the others without a question ({GUIDE[0]}, section 1)", self.again)
        if kind == "ring":
            head, workers = d["head"]["lan"]["addr"], self.trace_ring()
        else:
            head = (d.get("fabric") or {}).get("head_addr") or d["head"]["lan"]["addr"]
            hosts = [h for h in d.get("hosts", []) if h.get("access") == "ok" and h.get("gb10") and not h.get("other")]
            workers = [x for x in want if x != head] or [h["addr"] for h in hosts]
            self.fit_workers(workers, d, hosts)
        c = self.ready_nodes(head, workers)
        code, r = self.call("POST", "/v1/launch/config", {"head": head, "workers": c["workers"]})
        if code >= 300:
            raise Stop(f"the nodes could not be registered ({code}): {self.why(r)}", self.log_fix(), self.again)
        t = c.get("topology", {})
        cf = self.call("GET", "/v1/launch/config")[1]
        self.world = cf["world"]
        say(f"      registered: {t.get('mode')}, {' '.join(cf['nodes'])} (rank 0 first), in {cf.get('node_file')}")

    def ready_nodes(self, head, workers, registered=False):
        """The node check, passed; the image fetched once where it is what failed. On registered nodes the memory is
        left to the plan: the check's row is an estimate from glm53.conf, the plan's from the width and the KV asked."""
        for attempt in (1, 2):
            c = self.check(head, workers)
            if c.get("nodes_ok"):
                return c
            bad = [(n, i) for n, i in self.failed_items(c) if not (registered and i["key"] == "memory")]
            if registered and not bad:
                return c
            need = sorted({n["addr"] for n, i in bad if i["key"] == "image"})
            others = [(n, i) for n, i in bad if i["key"] != "image"]
            if others or not need or attempt == 2:
                self.stop_nodes(c, bad)
            self.get_image(need)

    def trace_ring(self):
        """The other three nodes, from the head along the cables (LLDP)."""
        code, r = self.call("POST", "/v1/launch/ring", {})
        if code not in (200, 202):
            raise Stop(f"the ring trace was refused ({code}): {self.why(r)}", self.log_fix(), self.again)
        t = self.wait("/v1/launch/ring", lambda x: x.get("state") == "done", "the ring trace")
        if t.get("ring") != "closed" or len(t.get("nodes", [])) != 4 or not all(t["nodes"]):
            ps, names = t.get("problems", []), " ".join(t.get("names", []))
            # a problem naming no node is the trace's own: this node's LAN
            bad = [f"  {p.get('name') or p.get('addr')}: {ACCESS_WHY.get(p.get('what'), p.get('what'))}" + (f": {p['detail']}" if p.get("detail") and p.get("what") != "noaddr" else "")
                   if p.get("name") or p.get("addr") else f"  {p.get('detail') or p.get('what')}" for p in ps]
            fix = f"a ring is four nodes, each node's port 0 cabled to the next one's port 1 (RING.md); this node must ssh to itself and to the others without a question ({GUIDE[0]}, section 1)"
            if any(p.get("what") == "open" for p in ps) or not ps:
                fix += "; every node needs lldpd running, which finds the neighbours (lldpctl shows them)"
            if any(p.get("what") == "noaddr" for p in ps):
                fix = f"from this node, ping each other node's LAN address, which puts it in this node's ARP table. If it still stops: {fix}"
            if any(not (p.get("name") or p.get("addr")) for p in ps):
                fix = "this node's LAN is the interface of its default route, which needs an IPv4 address; or write glm53.nodes.conf by hand (RING.md, Settings)"
            raise Stop(("no ring found" if t.get("ring") == "none" else f"the ring is {t.get('ring')}") + (f", nodes {names}" if names else "") + ":\n" + "\n".join(bad or ["  (no problem named)"]), fix, self.again)
        say(f"      ring: {' '.join(t['names'])}")
        return t["nodes"][1:]

    def fit_workers(self, workers, d, hosts):
        n = len(workers)
        if n + 1 in (2, 4, 8):
            return
        listed = ", ".join(f"{h['addr']} ({h.get('name') or '?'})" for h in hosts) or "none"
        kept = [h for h in d.get("hosts", []) if h.get("access") != "ok"]
        why = "".join(f"\n  {h['addr']}: {ACCESS_WHY.get(h.get('access'), h.get('access'))}" for h in kept)
        fix = "choose them: add -e NODES=\"ADDR ADDR ADDR\" with 3 of these addresses (7 for eight nodes) to the docker run line"
        if kept:
            fix = f"ssh from this node to each address above, and to its own ({(d.get('fabric') or {}).get('head_addr') or 'its ConnectX-7 address'}), must work without a question ({GUIDE[0]}, section 1). Then, if more are free than a lane uses: " + fix
        raise Stop(f"{n} free worker nodes found on {(d.get('fabric') or {}).get('subnet', 'the fabric')}: {listed}. A lane has 2, 4 or 8 nodes with this one" + why, fix, self.again)

    def check(self, head, workers):
        code, r = self.call("POST", "/v1/launch/check", {"head": head, "workers": workers, "deploy": True})
        if code not in (200, 202):
            raise Stop(f"the node check was refused ({code}): {self.why(r)}", self.log_fix(), self.again)
        say("      checking every node (copies this directory to them, then memory, GPU, RDMA, the image)")
        return self.wait("/v1/launch/check", lambda x: x.get("state") == "done", "the node check", every=3)

    @staticmethod
    def failed_items(c):
        return [(n, i) for n in c.get("nodes", []) for i in n.get("items", []) if i.get("state") == "fail" and i.get("key") not in ("weights", "banks")]

    def stop_nodes(self, c, bad):
        lines = [f"  {n.get('name') or n['addr']}: {i['key']}: {i.get('value', '')}" + (f" -> {i['fix']}" if i.get("fix") else "") for n, i in bad]
        t = c.get("topology") or {}
        if t.get("why"):
            lines.append(f"  cabling: {t['why']}")
            if "not read" in t["why"]:
                lines.append(f"  this node must ssh to itself and to every node without a question ({GUIDE[0]}, section 1)")
        why = c.get("why") or []
        if not bad:
            lines += [f"  {x}" for x in (why if isinstance(why, list) else [why])]
        raise Stop("a node is not ready:\n" + "\n".join(lines or ["  (no item failed; read the launcher's log)"]), "do what each line says", self.again)

    def get_image(self, need):
        """The image on the nodes that lack it or hold another build: docker pull, each at once; else how to copy it."""
        image = self.a.image or self.conf_value("IMAGE")
        say(f"      image {image}: docker pull on {' '.join(need)}")
        ps = [(n, subprocess.Popen(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", n, f"docker pull {image}"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)) for n in need]
        failed = []
        for n, p in ps:
            _, err = p.communicate()
            if p.returncode != 0:
                failed.append((n, err.strip().splitlines()[-1][:240] if err.strip() else f"rc {p.returncode}"))
        if failed:
            copy = "\n".join(f"  docker save {image} | ssh {n} docker load" for n, _ in failed)
            raise Stop(f"{image} is not on {' '.join(n for n, _ in failed)} and docker pull failed: {failed[0][1]}",
                       f"copy it from this node:\n{copy}", self.again)
        say("      image: pulled")

    def conf_value(self, key):
        m = None
        for line in open(self.a.conf):
            x = re.match(rf"\s*{key}=(\S*)", line)
            if x:
                m = x.group(1).strip("\"'")
        return m or ""

    # step 3 -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def home(p):
        return os.path.expanduser(p or "")

    def weights(self):
        w = self.call("GET", "/v1/launch/weights")[1]
        ck = self.opts.get("WEIGHTS") or w["weights"]
        if "WEIGHTS" not in self.opts and self.world == 8 and "-4x-" in ck:
            ck = ck.replace("-4x-", "-8x-")
        asked = {BANK_OPTIONS[k]: v for k, v in self.opts.items() if k in BANK_OPTIONS}
        unknown = sorted(set(asked) - {b["bits"] for b in w["banks"]})
        if unknown:
            raise Stop(f"BANK_{unknown[0]}: no {unknown[0]} bit row in expert-plan/banks.json", "use widths " + ", ".join(str(b["bits"]) for b in w["banks"]))
        drafter = self.opts.get("DRAFTER", w.get("drafter", ""))
        body = {"weights": ck, "drafter": drafter, "banks": [{"id": b["id"], "dir": asked.get(b["bits"], b["dir"])} for b in w["banks"]]}
        code, r = self.call("POST", "/v1/launch/weights/check", body)
        if code not in (200, 202):
            raise Stop(f"the weights check was refused ({code}): {self.why(r)}", "WEIGHTS, DRAFTER and BANK_N take absolute paths or ~/ paths", self.again)
        say("[3/6] weights: looking for the checkpoint, the 3-bit bank and the others on every node")
        v = self.wait("/v1/launch/weights", lambda x: x.get("check", {}).get("state") == "done", "the weights check", every=3)["check"]
        rows = {x["key"]: x for x in v.get("items", [])}
        nodes = ", ".join(n.get("name") or n["addr"] for n in v.get("nodes", []))

        def lacking(key):
            """Where this row is not there: its nodes, and the first one's reason."""
            its = [(n.get("name") or n["addr"], next((i for i in n.get("items", []) if i["key"] == key), {})) for n in v.get("nodes", [])]
            bad = [(a, i) for a, i in its if i.get("state") != "ok"]
            return f"not on {', '.join(a for a, _ in bad)} ({bad[0][1].get('value', '')})" if bad else "not there"

        dirs = {b["id"]: self.home(asked.get(b["bits"], b["dir"])) for b in w["banks"]}
        asked_bits = int(self.opts["BITS"]) if "BITS" in self.opts else None
        problems, optional = [], []
        if rows.get("weights", {}).get("state") == "fail":
            problems.append(("checkpoint (always needed)", self.home(ck), CHECKPOINT_REPO.get(self.world, "the checkpoint"), lacking("weights")))
        for b in w["banks"]:
            state = rows.get("bank:" + b["id"], {}).get("state")
            if b["bits"] == DEFAULT_BITS and state != "ok":
                problems.append((f"{b['bits']} bit (always needed)", dirs[b["id"]], b.get("repo", ""), lacking("bank:" + b["id"])))
            elif state == "fail" or (state == "warn" and b["bits"] == asked_bits):
                problems.append((f"{b['bits']} bit" + (" (BITS asks for it)" if b["bits"] == asked_bits else ""), dirs[b["id"]], b.get("repo", ""), lacking("bank:" + b["id"])))
            elif state == "warn":
                optional.append(b)
        if problems:
            self.stop_missing(problems, nodes, optional, dirs)
        for k, x in rows.items():
            if x.get("state") == "fail":
                raise Stop(f"{k}: {x.get('value')}", x.get("fix") or "put that directory right on the node named", self.again)
        changed = ck != w["weights"] or drafter != w.get("drafter", "") or any(asked.get(b["bits"], b["dir"]) != b["dir"] for b in w["banks"])
        if changed:
            code, r = self.call("POST", "/v1/launch/weights", body)
            if code >= 300:
                raise Stop(f"the weights could not be saved ({code}): {self.why(r)}", self.log_fix(), self.again)
        st = self.wait("/v1/launch/status", lambda x: not (x.get("banks") or {}).get("checking"), "the banks check", every=3)
        avail = sorted(int(k) for k, x in (st.get("banks", {}).get("bits") or {}).items() if x.get("ok"))
        say(f"      checkpoint: {self.home(ck)}: on every node. Expert banks with data: " + ", ".join(f"{x} bit" for x in avail))
        if optional:
            say("      optional, not used (for BITS or per-expert widths): " + "; ".join(f"{b['bits']} bit, {rows.get('bank:' + b['id'], {}).get('value', 'not there')}" for b in optional))

    def stop_missing(self, problems, nodes, optional, dirs):
        lines, fix = [], []
        for what, where, repo, why in problems:
            lines.append(f"  {what}: {where}: {why}")
            lines.append(f"    put {repo or 'its files'} there, the same path on every node ({nodes})")
            if repo:
                fix.append(f"  hf download {repo} --local-dir {where}")
        for b in optional:
            lines.append(f"  optional: {b['bits']} bit: {dirs[b['id']]}: not there ({b.get('repo', '')}); only for BITS={b['bits']} or per-expert widths")
        raise Stop("the model's files are not on the nodes:\n" + "\n".join(lines),
                   f"on each node, or copy from a node that has them ({self.pkg}/{GUIDE[0]}: the downloads); already elsewhere: add -e WEIGHTS=DIR or -e BANK_3=DIR to the docker run line; another width you do not use: leave it out with -e BANK_N=\n" + "\n".join(fix),
                   self.again)

    # step 4, 5 -----------------------------------------------------------------------------------------------------------
    def apply_settings(self):
        """The engine settings given, saved in glm53.conf as the settings screen saves them (checked there): after the
        plan passed, so a start that cannot fit leaves the settings as they were."""
        keys = {k: v for k, v in self.opts.items() if k not in OWN_OPTIONS and k != "KV_SIZE"}
        changed = {k: v for k, v in keys.items() if str(self.settings[k].get("value", "")) != v}
        if not changed:
            return
        code, r = self.call("POST", "/v1/launch/settings", {"values": changed})
        if code >= 300:
            errs = r.get("errors") or {}
            raise Stop(f"settings refused ({code}): " + (("\n" + "\n".join(f"  {k}: {x}" for k, x in errs.items())) if errs else str(self.why(r))),
                       f"each key's rule: {self.pkg}/CONFIG.md, or GET /v1/launch/settings", self.again)
        say("      saved in glm53.conf: " + ", ".join(f"{k}={v}" for k, v in changed.items()))

    def asked(self):
        """The options a plan takes itself, so it can be judged before they are saved."""
        out = {}
        for key, name, conv in (("KV", "kv", str), ("DRAFTER_KIND", "drafter", str), ("CONTEXT_LENGTH", "context", int),
                                ("RESERVE_GB", "reserve_gib", float), ("MULTI", "multi", int), ("MULTI_WAIT", "multi_wait", int)):
            if key in self.opts:
                try:
                    out[name] = conv(self.opts[key])
                except ValueError:
                    raise Stop(f"{key}={self.opts[key]}: not a {conv.__name__}", f"each key's rule: {self.pkg}/CONFIG.md", self.again)
        return out

    def choose(self):
        """The plan: an allocation, or every expert at one width, 3 bits when none is given."""
        o = {}
        kv_size = self.opts.get("KV_SIZE", "max")
        if kv_size == "max":
            o["store"] = "max"
        elif kv_size.isdigit():
            context = int(self.opts.get("CONTEXT_LENGTH") or self.settings["CONTEXT_LENGTH"]["value"])
            if int(kv_size) < context + 512:
                raise Stop(f"KV_SIZE={kv_size} is too small: the KV holds the context length, {context} tokens, and 512 more", "give a larger KV_SIZE, or a smaller CONTEXT_LENGTH with -e CONTEXT_LENGTH=N", self.again)
            o["store"] = int(kv_size) - context
        else:
            raise Stop(f"KV_SIZE={kv_size}: a number of tokens, or max", f"each key's rule: {self.pkg}/CONFIG.md", self.again)
        if "ALLOC" in self.opts:
            name = self.opts["ALLOC"]
            code, al = self.call("GET", f"/v1/launch/alloc?name={urllib.parse.quote(name)}")
            if code != 200:
                names = [x["name"] for x in self.call("GET", "/v1/launch/allocs")[1].get("allocs", []) if x.get("valid")]
                raise Stop(f"ALLOC={name}: {self.why(al)}", "saved allocations here: " + (", ".join(names) or "none"), self.again)
            bits, start, what = al["bits"], dict(o, alloc=name), f"the allocation {name}"
        else:
            w = int(self.opts.get("BITS", DEFAULT_BITS))
            bits = base64.b64encode(bytes([w]) * (LAYERS * EXPERTS)).decode()
            start, what = dict(o, bits=bits), f"every expert at {w} bit"
        code, p = self.call("POST", "/v1/launch/plan", dict(o, bits=bits, **self.asked()))
        if code != 200:
            raise Stop(f"the plan was refused ({code}): {self.why(p)}", "check KV, DRAFTER_KIND, MULTI and CONTEXT_LENGTH on the docker run line", self.again)
        kv = p["kv"]
        margin = min(n["margin"] for n in p["nodes"]) / (1 << 30) if p.get("nodes") else 0
        if p["verdict"] == "no":
            why = [f"{w} bit: {x}" for w, x in (p.get("missing_bits") or {}).items()]
            if (p.get("multi") or {}).get("why"):
                why.append(p["multi"]["why"])
            if not why:
                why.append(f"{margin:.1f} GiB would be left on the fullest node")
            room = f"context length {kv['context']}" if kv["context"] else f"any KV cache in {kv['format']}"
            raise Stop(f"{what} does not fit with {room}: " + "; ".join(why),
                       "a smaller CONTEXT_LENGTH, a narrower BITS (3 fits four nodes), KV=fp4, no DFlash2, a lower MULTI, or stop what else uses the nodes' memory (free -g). Nothing was saved", self.again)
        if kv["formats"][kv["format"]].get("cut"):
            self.notes.append(f"the context length was cut to {kv['context']} tokens to fit (asked {kv.get('context_asked') or kv.get('conf_context')})")
        if p["verdict"] == "tight":
            self.notes.append(f"only {margin:.1f} GiB left on the fullest node (RESERVE_GB is {kv.get('reserve_gib', p.get('reserve_gib'))})")
        say(f"[4/6] plan: {what}, KV {kv['format']}, context length {kv['context']} tokens, KV size {kv['context'] + kv['store']} tokens, {margin:.1f} GiB to spare")
        return start

    def start(self, body):
        code, r = self.call("POST", "/v1/launch/start", body)
        if code >= 300:
            raise Stop(f"the start was refused ({code}): {self.why(r)}", self.log_fix(), self.again)
        say("[5/6] start: the engine is loading (minutes; the dashboard shows it too)")
        t0 = last = time.time()
        while True:
            time.sleep(10)
            st = self.call("GET", "/v1/launch/status")[1]
            s = st.get("state")
            if s == "running" and not st.get("phase"):
                say(f"      running after {int(time.time() - t0)} s")
                return
            if s in ("failed", "stopped"):
                raise Stop("the engine did not start: " + (st.get("error") or s), f"read the logs: cd {self.pkg} && ./tf-glm53 logs launcher; ./tf-glm53 logs 0", self.again)
            if time.time() - last >= 30:
                last = time.time()
                eta = st.get("eta_ms") or 0
                say(f"      {int(last - t0)} s: {s} {st.get('phase', '')} layers {st.get('layers_done', 0)}/{st.get('layers', LAYERS)}" + (f", about {eta // 1000} s left (estimate)" if eta else ""))

    # step 6 -------------------------------------------------------------------------------------------------------------
    def api(self, path, body=None, timeout=180):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.ep}{path}", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def smoke(self):
        model = self.api("/v1/models")["data"][0]["id"]
        j = self.api("/v1/chat/completions", {"model": model, "max_tokens": 4000, "messages": [{"role": "user", "content": "What is 6 times 6? Answer with the number only."}]})
        msg = j["choices"][0]["message"]
        ok = "36" in (msg.get("content") or "")
        say(f"[6/6] inference: {'answered 36' if ok else 'unexpected answer: ' + repr((msg.get('content') or '')[:80])}, {j['usage']['total_tokens']} tokens")
        if not ok:
            raise Stop("the engine runs but its answer is wrong", self.log_fix("0"))
        return model

    def where(self):
        return [f"  dashboard (no key)  http://{self.lan}:{self.lp}/panel",
                f"  control API         http://{self.lan}:{self.lp}/v1/launch/status",
                f"  first start         {self.pkg}/{GUIDE[0]}",
                f"  guide for AI        {self.pkg}/{GUIDE[1]} (control API), {self.pkg}/{GUIDE[2]} (the whole system)"]

    def ready(self, model):
        say()
        say("READY")
        say(f"  inference API       http://{self.lan}:{self.ep}/v1 (OpenAI compatible, no key, model {model})")
        for x in self.where():
            say(x)
        say(f"  try it              curl -s http://{self.lan}:{self.ep}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'")
        say(f"  stop                the dashboard, or {self.pkg}/{GUIDE[1]} section 7 (saves the usage first, then stops)")
        for n in self.notes:
            say("  NOTE                " + n)
        say("  note                anyone who can reach these ports can use them: keep the nodes on a closed network")

    def run(self):
        try:
            self.launcher()
            self.check_options()
            self.nodes()
            st = self.call("GET", "/v1/launch/status")[1]
            if st.get("state") == "running" and not st.get("phase"):
                say("[3/6] the engine runs already: nothing started")
                if self.opts:
                    self.notes.append("options not applied, the engine runs (stop it, then run this again): " + ", ".join(self.opts))
                self.ready(self.smoke())
                return 0
            if st.get("state") not in ("stopped", "failed"):
                raise Stop(f"the engine is {st.get('state')} {st.get('phase', '')}", "wait until it is running (the dashboard shows it), or stop it", self.again)
            self.weights()
            body = self.choose()
            self.apply_settings()
            self.start(body)
            self.ready(self.smoke())
            return 0
        except Stop as e:
            say()
            say("STOP: " + e.what)
            say("FIX:  " + e.fix)
            say("THEN: " + (e.then or self.again))
            if self.listening:
                say("Meanwhile the launcher stays up:")
                for x in self.where():
                    say(x)
            return 2
        except (OSError, ValueError, KeyError) as e:
            say()
            say(f"STOP: quickstart failed: {e!r}")
            say("FIX:  " + self.log_fix())
            say("THEN: " + self.again)
            return 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkg", required=True)
    p.add_argument("--conf", required=True)
    p.add_argument("--launcher-port", type=int, required=True)
    p.add_argument("--api-port", type=int, required=True)
    p.add_argument("--again", default="", help="the command to run after a fix (docker run ... up)")
    p.add_argument("--image", default="")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="an option (above)")
    sys.exit(Quick(p.parse_args()).run())


if __name__ == "__main__":
    main()
