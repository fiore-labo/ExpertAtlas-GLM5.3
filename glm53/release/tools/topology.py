#!/usr/bin/env python3
"""topology.py: how a lane's nodes are cabled, judged once from what each node says (tf-glm53 topology; TOPOLOGY=auto).

Each node of NODES is read over ssh, in parallel: its ConnectX-7 halves (/sys/class/infiniband/roce*: the net device,
its state, its IPv4 address, the index of its RoCE v2 GID with an IPv4 address), the LLDP neighbours on those net
devices (lldpctl), its interfaces' IPv4 addresses and the sha256 of NCCL_LIB. The judgment:

  switch    no node has a GB10 neighbour by LLDP, and there is a switch by LLDP (a neighbour whose port is not a
            ConnectX-7 net device) or a /24 three or more nodes share: mode full on the half every node has on the
            /24 most nodes are on (a tie: IB_HCA's, then the first PCIe halves')
  ring4     four nodes, out of the head's port 0 into the next node's port 1 (crossed) and so on back to the head,
            both lanes of every link up on two /24s of their own, the opposite pairs sharing none: mode ring4, NODES in
            ring order (the head, rank 0, kept)
  direct    two nodes on one cable, or three closed the same way (a triangle): mode full
  none      anything else (a node not read, a ring not closed, a lane down, ...): no mode, nothing falls back
  conflict  LLDP says GB10s cabled to each other while every address is in one /24

An explicit TOPOLOGY (full or ring4) wins; the judgment then says whether the cabling agrees. --json prints the whole
judgment (the control panel's GET /v1/launch/topology), --shell the lines tf-glm53 sets its settings from, and the
default a few lines for a person. Problems carry English sentences (detail, fix) and their values (rank, name, port,
half, subnet, other) for a screen to word its own way.

The addresses alone decide when LLDP is silent (lldpd not installed): a /24 two nodes share is a cable between them,
the far end of a port is the one other node on its /24. LLDP, where it answers, names the far end instead and the
switch's port. Two things the hardware shows over LLDP and the rule allows for: a node sees itself (the two PCIe halves
of a port share one cable), and a ring's neighbour shows both of its halves on each of ours.
"""
import argparse
import ipaddress
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PROBE = r'''echo "host=$(hostname)"
for d in /sys/class/infiniband/roce*; do
  [ -d "$d" ] || continue; h=${d##*/}; n=$(ls $d/device/net 2>/dev/null | head -1); [ -n "$n" ] || continue
  g=-; for i in $(seq 0 15); do t=$(cat $d/ports/1/gid_attrs/types/$i 2>/dev/null); a=$(cat $d/ports/1/gids/$i 2>/dev/null)
    case "$t:$a" in 'RoCE v2:0000:0000:0000:0000:0000:ffff:'*) g=$i ;; esac; done
  echo "dev=$h $n $(cat /sys/class/net/$n/operstate) $(ip -4 -o addr show dev $n | awk '{print $4; exit}' | grep . || echo -) $g"
done
ip -4 -o addr show | awk '{print "ip=" $2 " " $4}'
echo "nccl=$(sha256sum %s 2>/dev/null | cut -c1-64)"
lldpctl -f keyvalue 2>/dev/null | grep -E '^lldp\.[^.]+\.(via|chassis\.name|port\.descr)='
'''
CX7_NET = re.compile(r"^en[pP].*s0f[01]np[01]$")


def port_of(name):
    """p0 / p1 of a ConnectX-7 device or net device (rocep1s0f1, enP2p1s0f1np1: p1), "" for another name."""
    m = re.search(r"s0f(\d)", name)
    return f"p{m.group(1)}" if m else ""


def half_of(name):
    """The PCIe half of a ConnectX-7 device or net device: 1 for the second (roceP2p1s0f0, enP2p1s0f0np0), else 0."""
    return 1 if re.search(r"P2p1", name) else 0


def net24(addr):
    """The /24 of "192.0.2.10/24" ("192.0.2.0/24"), "" for none."""
    if not addr or addr == "-":
        return ""
    return str(ipaddress.ip_interface(addr.split("/")[0] + "/24").network)


def access_of(stderr):
    e = stderr or ""
    if "Host key verification failed" in e or "REMOTE HOST IDENTIFICATION HAS CHANGED" in e:
        return "hostkey"
    if "Permission denied" in e:
        return "nokey"
    return "noanswer"


def read(addr, nccl_lib):
    """One node over ssh: {addr, name, access, halves[], ifaddrs{}, nccl}."""
    n = {"addr": addr, "name": "", "access": "noanswer", "halves": [], "ifaddrs": {}, "nccl": ""}
    lib = "'" + nccl_lib.replace("'", "'\\''") + "'" if nccl_lib else "/dev/null"
    try:
        r = subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", addr, PROBE % lib],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return n
    if r.returncode == 255:
        n["access"] = access_of(r.stderr)
        return n
    devs, nb, cur = [], {}, None
    for line in r.stdout.splitlines():
        if line.startswith("host="):
            n["name"] = line[5:].strip()
        elif line.startswith("dev="):
            devs.append(line[4:].split())
        elif line.startswith("ip="):
            f = line[3:].split()
            if len(f) == 2:
                n["ifaddrs"].setdefault(f[0], []).append(f[1])
        elif line.startswith("nccl="):
            n["nccl"] = line[5:].strip()
        elif line.startswith("lldp."):
            k, _, v = line.partition("=")
            p = k.split(".", 2)
            if len(p) < 3:
                continue
            if p[2] == "via":
                cur = {}
                nb.setdefault(p[1], []).append(cur)
            elif cur is not None and nb.get(p[1]) and nb[p[1]][-1] is cur:
                cur[{"chassis.name": "name", "port.descr": "port"}[p[2]]] = v.strip()
    n["access"] = "ok"
    for f in devs:
        if len(f) < 5:
            continue
        dev, net, state, addr4, gid = f[:5]
        peer = None
        others = [x for x in nb.get(net, []) if x.get("name") and x.get("name") != n["name"]]
        gb = [x for x in others if CX7_NET.match(x.get("port", ""))]
        if gb:
            peer = {"kind": "gb10", "name": gb[0]["name"], "port": port_of(gb[0]["port"])}
        elif others:
            peer = {"kind": "switch", "name": others[0]["name"], "port": others[0].get("port", "")}
        n["halves"].append({"dev": dev, "net": net, "port": port_of(dev), "half": half_of(dev), "up": state == "up",
                            "addr": "" if addr4 == "-" else addr4, "gid": None if gid == "-" else int(gid), "peer": peer})
    n["halves"].sort(key=lambda h: (h["port"], h["half"]))
    return n


def judge(a):
    nodes = a.nodes
    t0 = time.time()
    if a.readings:  # what the nodes said, kept by --dump (the tests change it to cablings not at hand)
        with open(a.readings) as f:
            rd = json.load(f)
    else:
        with ThreadPoolExecutor(len(nodes)) as ex:
            rd = list(ex.map(lambda x: read(x, a.nccl_lib), nodes))
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump(rd, f, indent=1)
    probs = []

    def problem(what, rank=None, detail="", fix="", **kw):
        p = {"what": what, "rank": rank, "addr": nodes[rank] if rank is not None else kw.pop("addr", ""),
             "name": rd[rank]["name"] if rank is not None else kw.pop("name", ""), "detail": detail, "fix": fix}
        p.update({k: kw.get(k) for k in ("port", "half", "subnet", "other")})
        probs.append(p)

    def nm(i):
        return rd[i]["name"] or nodes[i]

    for i, n in enumerate(rd):
        if n["access"] != "ok":
            fix = {"hostkey": f"ssh {nodes[i]} once from this node and accept its host key",
                   "nokey": f"this node's ssh key in {nodes[i]}'s authorized_keys",
                   "noanswer": f"the node {nodes[i]} up and reachable over ssh"}[n["access"]]
            problem(n["access"], i, f"{nodes[i]}: not read over ssh ({n['access']})", fix)
    ok = [i for i, n in enumerate(rd) if n["access"] == "ok"]
    rank_of = {rd[i]["name"]: i for i in ok}

    def up(i):
        return [h for h in rd[i]["halves"] if h["up"]]

    def gb(i):
        return [h for h in up(i) if h["peer"] and h["peer"]["kind"] == "gb10"]

    def sw(i):
        return [h for h in up(i) if h["peer"] and h["peer"]["kind"] == "switch"]

    def head_if():
        """The interface rank 0 holds NODES[0]'s address on (the rendezvous, SOCKET_IFNAME), "" if none does."""
        if rd[0]["access"] != "ok":
            return ""
        return next((f for f, ads in rd[0]["ifaddrs"].items() if any(x.split("/")[0] == nodes[0] for x in ads)), "")

    # a GB10 neighbour no node read calls itself: the one node not read, when one was not (its name from LLDP), else a
    # node that is not one of NODES (its name only, no address to read it by)
    unknown = sorted({h["peer"]["name"] for i in ok for h in gb(i)} - set(rank_of))
    unread_ranks = [i for i, n in enumerate(rd) if n["access"] != "ok"]
    if len(unknown) == 1 and len(unread_ranks) == 1:
        rank_of[unknown[0]] = unread_ranks[0]
        rd[unread_ranks[0]]["name"] = unknown[0]
    for i in ok:
        for h in gb(i):
            if unread_ranks:
                break
            if h["peer"]["name"] not in rank_of and not any(p["what"] == "noaddr" and p["name"] == h["peer"]["name"] for p in probs):
                problem("noaddr", None, f"{nm(i)} {h['port']}: LLDP names {h['peer']['name']}, which is not in NODES",
                        f"add {h['peer']['name']}'s LAN address to NODES, or move the cable", name=h["peer"]["name"],
                        port=h["port"], other=i)
    out = {"verdict": "none", "mode": None, "why": "", "order": list(nodes), "links": [], "opposite": [], "fabric": None,
           "closed": False, "hca": "", "ifname": ""}
    unread = len(ok) < len(nodes)

    # who holds each /24 on a half that is up: a /24 of two nodes is a cable between them, of three or more a switch.
    # The addresses decide without LLDP (lldpd not installed); LLDP, where it answers, names the far end and the switch
    holders = {}
    for i in ok:
        for h in up(i):
            if h["addr"]:
                holders.setdefault(net24(h["addr"]), set()).add(i)
    lldp_gb, lldp_sw = any(gb(i) for i in ok), any(sw(i) for i in ok)
    wide = sorted(s for s, hs in holders.items() if len(hs) >= 3)
    pairs = [s for s, hs in holders.items() if len(hs) == 2]
    common = [s for s, hs in holders.items() if hs == set(ok)]

    def cand(i):
        """A node's halves toward a switch: behind one by LLDP, else (LLDP silent) every half up with an address."""
        return sw(i) or [h for h in up(i) if h["addr"] and not h["peer"]]

    # a switch: no GB10 neighbour by LLDP, and a switch by LLDP, or by the addresses a /24 three or more nodes share
    # and no /24 of just two (a cable); two nodes on one /24 the same way
    if ok and not lldp_gb and (lldp_sw or (len(nodes) == 2 and common) or (len(nodes) > 2 and wide and not pairs)):
        want = a.hca.split(",")[0] if a.hca and "," not in a.hca else ""

        def pick(i, sub=None):
            hs = [h for h in cand(i) if h["addr"] and (sub is None or net24(h["addr"]) == sub)]
            return next((h for h in hs if h["dev"] == want), hs[0] if hs else None)

        counts = {}
        for i in ok:
            for s in {net24(h["addr"]) for h in cand(i) if h["addr"]}:
                counts[s] = counts.get(s, 0) + 1
        # the /24 most nodes have a switch half on; a tie goes to IB_HCA's, then to the first PCIe halves'
        mine = {net24(h["addr"]) for i in ok for h in cand(i) if h["dev"] == want and h["addr"]}
        first = {net24(h["addr"]) for i in ok for h in cand(i) if h["half"] == 0 and h["addr"]}
        major = max(counts, key=lambda s: (counts[s], s in mine, s in first, [-int(x) for x in s.split("/")[0].split(".")])) if counts else ""
        members = []
        for i in ok:
            h = pick(i, major) or pick(i)
            members.append({"rank": i, "dev": h["dev"] if h else "", "addr": h["addr"] if h else "",
                            "port": h["peer"]["port"] if h and h["peer"] else "", "switch": h["peer"]["name"] if h and h["peer"] else "",
                            "net": h["net"] if h else ""})
            if not h or net24(h["addr"]) != major:
                problem("split", i, f"{nm(i)} {h['dev'] if h else ''}: {h['addr'] if h else 'no address'} is not in {major}",
                        f"an address in {major} on {h['net'] if h else 'its switch half'}", port=h["port"] if h else None,
                        half=h["half"] if h else None, subnet=net24(h["addr"]) if h else "", other=major)
        split = any(p["what"] == "split" for p in probs)
        out["fabric"] = {"subnet": major, "split": split, "switches": sorted({m["switch"] for m in members if m["switch"]}),
                         "members": [{k: m[k] for k in ("rank", "dev", "addr", "port")} for m in members]}
        devs = sorted({m["dev"] for m in members if m["dev"]})
        out["hca"] = devs[0] if len(devs) == 1 else ""
        h0 = next((h for h in rd[0]["halves"] if h["dev"] == out["hca"]), None) if rd[0]["access"] == "ok" else None
        out["ifname"] = head_if() or (h0["net"] if h0 else "")
        if unread:
            out["why"] = "a node not read"
        elif split:
            out["why"] = "the switch halves are not on one /24"
        elif len(devs) > 1:
            # IB_HCA names one device for every rank: the nodes off the device most of them use
            ref = max(devs, key=lambda d: (sum(m["dev"] == d for m in members), d == want, -devs.index(d)))
            by = next(m["rank"] for m in members if m["dev"] == ref)
            for m in members:
                if m["dev"] and m["dev"] != ref:
                    problem("ports", m["rank"], f"{nm(m['rank'])}: {m['dev']} on {major}, {ref} on {nm(by)}",
                            f"the cable in {port_of(ref)} of {nm(m['rank'])}, as on the others ({ref})", port=port_of(m["dev"]),
                            half=half_of(m["dev"]), subnet=major, other=by)
            out["why"] = "the nodes reach the /24 on different devices: IB_HCA names one for every rank"
        else:
            out.update(verdict="direct" if len(nodes) == 2 and not lldp_sw else "switch", mode="full")
        return finish(a, out, rd, probs, t0)

    # GB10s cabled to each other: the walk out of the head's port 0
    def nb(i, port):
        """The far end of node I's PORT: LLDP's GB10, else the one other node with a half on the same /24."""
        if i is None or rd[i]["access"] != "ok":
            return None
        hs = [h for h in rd[i]["halves"] if h["port"] == port]
        for h in hs:
            if h["peer"] and h["peer"]["kind"] == "gb10":
                return dict(h["peer"], via="lldp")
        for h in hs:
            js = holders.get(net24(h["addr"]), set()) - {i} if h["up"] and h["addr"] and not h["peer"] else set()
            if len(js) == 1:
                j = js.pop()
                y = next(x for x in up(j) if x["addr"] and net24(x["addr"]) == net24(h["addr"]))
                return {"kind": "gb10", "name": rd[j]["name"], "port": y["port"], "via": "addresses"}
        # a /24 no other node holds, with one node not read: that node (its port: the crossing's)
        if len(unread_ranks) == 1 and any(h["up"] and h["addr"] and not h["peer"] and not holders.get(net24(h["addr"]), set()) - {i} for h in hs):
            u = unread_ranks[0]
            rank_of.setdefault(nodes[u], u)
            return {"kind": "gb10", "name": rd[u]["name"] or nodes[u], "port": "p1" if port == "p0" else "p0", "via": "addresses"}
        return None

    def lane_ok(i, port, half):
        return any(h["up"] and h["addr"] for h in rd[i]["halves"] if h["port"] == port and h["half"] == half)

    def addr(i, port, half):
        return next((h["addr"] for h in rd[i]["halves"] if h["port"] == port and h["half"] == half), "")

    def link(x, px, y, py, via):
        rb = rd[y]["access"] == "ok"
        lanes, subs = [], []
        for hh in (0, 1):
            s = net24(addr(x, px, hh))
            good = lane_ok(x, px, hh) and (not rb or (lane_ok(y, py, hh) and net24(addr(y, py, hh)) == s))
            lanes.append(good)
            subs.append(s)
            if not good:
                why = "down" if not lane_ok(x, px, hh) or (rb and not lane_ok(y, py, hh)) else "on two /24s"
                problem("half", x, f"{nm(x)} {px}.{hh} - {nm(y)} {py}.{hh}: the lane is {why}",
                        f"the cable {nm(x)} {px} - {nm(y)} {py} (one half), and one /24 on both ends of the lane",
                        port=px, half=hh, subnet=s, other=y)
        return {"a": x, "b": y, "pa": px, "pb": py, "lanes": lanes, "subnets": subs, "crossed": px != py, "one_side": not rb,
                "via": via}

    # a port with no GB10 on it: a problem unless two nodes on one cable (the other ports are free then)
    head, order, links, closed, opens = 0, [0], [], False, []
    if rd[0]["access"] == "ok":
        cur, port = 0, "p0"
        for _ in range(8):
            p = nb(cur, port)
            if not p:
                opens.append((cur, port))
                break
            nx = rank_of.get(p["name"])
            if nx is None:
                break
            links.append(link(cur, port, nx, p["port"], p["via"]))
            if nx == head:
                closed = True
                break
            if nx in order:
                break
            order.append(nx)
            if rd[nx]["access"] != "ok":
                break
            cur, port = nx, ("p0" if p["port"] == "p1" else "p1")
        if not closed:
            cur, port, back = 0, "p1", []
            for _ in range(8):
                p = nb(cur, port)
                if not p:
                    if cur != head:
                        opens.append((cur, port))
                    break
                nx = rank_of.get(p["name"])
                if nx is None:
                    break
                if len(order) > 1 and nx == order[-1]:
                    links.append(link(cur, port, nx, p["port"], p["via"]))
                    closed = True
                    break
                if nx in order:
                    break
                links.append(link(cur, port, nx, p["port"], p["via"]))
                back.append(nx)
                if rd[nx]["access"] != "ok":
                    break
                cur, port = nx, ("p0" if p["port"] == "p1" else "p1")
            order += back[::-1]
    pair = len(nodes) == 2 and len(order) == 2 and bool(links)
    for cur, port in ([] if pair else opens):
        problem("open", cur, f"{nm(cur)} {port}: no GB10 on it", f"the cable on {nm(cur)} {port}", port=port)
    # crossed cables are what a ring and a triangle are walked by; two nodes take one device name on both ends
    for lk in ([] if pair else links):
        if not lk["crossed"]:
            problem("uncrossed", lk["a"], f"{nm(lk['a'])} {lk['pa']} - {nm(lk['b'])} {lk['pb']}: the same port number on both ends",
                    "port 0 to the next node's port 1 (crossed)", port=lk["pa"], other=lk["b"])
    if len(order) == 4 and closed:
        for x, y in ((order[0], order[2]), (order[1], order[3])):
            if rd[x]["access"] != "ok" or rd[y]["access"] != "ok":
                out["opposite"].append({"a": x, "b": y, "shared": [], "unknown": True})
                continue
            sx = {net24(h["addr"]) for h in rd[x]["halves"] if h["addr"]}
            sy = {net24(h["addr"]) for h in rd[y]["halves"] if h["addr"]}
            shared = sorted(sx & sy)
            out["opposite"].append({"a": x, "b": y, "shared": shared, "unknown": False})
            for s in shared:
                problem("opposite", x, f"{nm(x)} and {nm(y)} (opposite) share {s}", "a /24 of its own on each link (RING.md)",
                        subnet=s, other=y)
    allsubs = sorted(holders)
    mixed = lldp_gb and len(ok) >= 3 and len(allsubs) == 1
    if mixed:
        problem("mixed", 0, f"LLDP shows GB10s cabled to each other, but every address is in {allsubs[0]}",
                "a /24 of its own on each link for a ring, or the cables to the switch", subnet=allsubs[0])
    out.update(order=order + [i for i in range(len(nodes)) if i not in order] if len(order) < len(nodes) else order,
               links=links, closed=closed)
    bad = [p for p in probs if p["what"] in ("uncrossed", "opposite", "half", "open")]
    if unread or any(p["what"] == "noaddr" for p in probs):
        out["why"] = "a node not read" if unread else "a neighbour not in NODES"
    elif mixed:
        probs[:] = [p for p in probs if p["what"] != "opposite"]
        out.update(verdict="conflict", why="LLDP says a ring, the addresses a switch")
    elif closed and len(order) == 4 and len(nodes) == 4 and not bad:
        out.update(verdict="ring4", mode="ring4")
        hcas = sorted({h["dev"] for h in rd[0]["halves"] if h["up"]}, key=lambda d: (port_of(d), half_of(d)))
        out["hca"] = ",".join(hcas)
        out["ifname"] = head_if()
    elif ((closed and len(order) == 3) or (len(order) == 2 and links)) and len(order) == len(nodes) and not bad:
        out.update(verdict="direct", mode="full")
        if pair:
            # IB_HCA names one device for every rank: a half of a cable from port N to port N, up on one /24
            h = next((h for lk in links if lk["pa"] == lk["pb"] for hh in (0, 1) if lk["lanes"][hh]
                      for h in rd[0]["halves"] if h["port"] == lk["pa"] and h["half"] == hh), None)
            out["hca"] = h["dev"] if h else ""
            out["ifname"] = head_if() or (h["net"] if h else "")
            if not h:
                lk = links[0]
                problem("ports", 0, f"{nm(0)} {lk['pa']} - {nm(1)} {lk['pb']}: two nodes on one link take the same device name on both ends",
                        f"the cable from {nm(0)} {lk['pa']} to {nm(1)} {lk['pa']}", port=lk["pa"], other=1)
                problem("ports", 1, f"{nm(1)} {lk['pb']} - {nm(0)} {lk['pa']}: the other end of that link",
                        f"the cable in {nm(1)} {lk['pa']}", port=lk["pb"], other=0)
                out.update(verdict="none", mode=None, why="two nodes cabled port 0 to port 1: IB_HCA names one device for both")
    else:
        out["why"] = (f"a ring of {len(order)} nodes, NODES has {len(nodes)}" if closed and len(order) != len(nodes)
                      else "the ring does not hold" if closed else "the ring is not closed")
    return finish(a, out, rd, probs, t0)


def finish(a, out, rd, probs, t0):
    nodes = a.nodes
    conf = a.conf
    verdict_mode = out["mode"]
    if conf in ("full", "ring4"):
        mode = conf
        agrees = verdict_mode == conf
        if not agrees:
            said = {"full": "a switch or one link", "ring4": "a ring"}[conf]
            seen = {"full": "a switch or one link", "ring4": "a ring"}.get(verdict_mode or "", f"no verdict ({out['verdict']})")
            probs.append({"what": "conf", "rank": None, "addr": "", "name": "", "port": None, "half": None, "subnet": None,
                          "other": verdict_mode, "detail": f"TOPOLOGY={conf} ({said}), the cabling: {seen}",
                          "fix": "TOPOLOGY=auto, or the TOPOLOGY the cabling has"})
    else:
        conf = "auto"
        mode = verdict_mode
        agrees = mode is not None
    order = out["order"] if mode == "ring4" and out["verdict"] == "ring4" else list(range(len(nodes)))
    res = {
        "state": "done", "judged_unix": int(time.time()), "seconds": round(time.time() - t0, 2),
        "conf": conf, "verdict": out["verdict"], "mode": mode, "agrees": agrees, "why": out["why"],
        "order": [nodes[i] for i in order], "names": [rd[i]["name"] or nodes[i] for i in order],
        "reorder": order != list(range(len(nodes))),
        "nodes": [{"rank": i, "addr": nodes[i], "name": n["name"], "access": n["access"],
                   "halves": [{k: h[k] for k in ("dev", "net", "port", "half", "up", "addr", "peer")} for h in n["halves"]]}
                  for i, n in enumerate(rd)],
        "links": out["links"], "opposite": out["opposite"], "fabric": out["fabric"], "problems": probs,
        "hca": out["hca"] if mode == verdict_mode else "", "ifname": out["ifname"] if mode == verdict_mode else "",
    }
    res["prereq"] = prereq(a, res, rd)
    res["multi"] = ({"max_lane": 4, "why": "a ring: 3 and 4 conversations at once not run on a ring yet"} if mode == "ring4"
                    else {"max_lane": 1, "why": "three nodes: no concurrent conversations"} if len(nodes) == 3
                    else {"max_lane": 4, "why": ""})
    return res


def prereq(a, res, rd):
    """The mode's prerequisites, a row a node (or the lane's, rank null): tf-glm53 check's own, by mode."""
    mode, items = res["mode"], []

    def it(key, rank, state, value, fix=""):
        items.append({"key": key, "rank": rank, "state": state, "value": value, "fix": fix if state != "ok" else ""})

    readable = [i for i, n in enumerate(rd) if n["access"] == "ok"]
    if mode == "ring4":
        it("dcp", None, "ok" if a.dcp <= 2 else "fail", str(a.dcp), "DCP=1 or 2 (a DCP group must be a cabled pair)")
        for i in readable:
            if a.runtime == "host":  # RUNTIME=docker: the image's (tf-glm53 check looks for it there)
                good = bool(a.nccl_sha256) and rd[i]["nccl"] == a.nccl_sha256
                it("nccl", i, "ok" if good else "fail", rd[i]["nccl"][:12] or "missing",
                   "NCCL_SHA256 set" if not a.nccl_sha256 else f"NCCL_LIB on {rd[i]['name']}: the fenced build whose sha256 is NCCL_SHA256 (RING.md)")
            g = sum(1 for h in rd[i]["halves"] if h["gid"] is not None)
            it("gid4", i, "ok" if g >= 4 else "fail", str(g), f"an IPv4 address (RoCE v2 GID) on each of {rd[i]['name']}'s four devices")
            lks = [lk for lk in res["links"] if lk["a"] == i or lk["b"] == i]
            down = sum(1 for lk in lks for x in lk["lanes"] if not x)
            it("lanes", i, "ok" if lks and not down else "fail", f"{2 * len(lks) - down}/{2 * len(lks)}", f"the cables on {rd[i]['name']}")
    elif mode == "full":
        fab = res["fabric"]
        if fab:
            it("subnet", None, "ok" if fab["subnet"] and not fab["split"] else "fail", fab["subnet"] or "-", "every node's switch half on one /24")
        elif res["verdict"] == "direct":
            it("subnet", None, "ok", "direct", "")
        hca = res["hca"] or (a.hca if res["verdict"] not in ("switch", "direct") else "")
        for i in readable:
            h = next((h for h in rd[i]["halves"] if hca and h["dev"] == hca), None)
            it("hca", i, "ok" if h and h["up"] and h["addr"] else "fail", hca or "-", f"IB_HCA: a device up with an address on {rd[i]['name']}")
            it("gid", i, "ok" if h and h["gid"] is not None else "fail", str(h["gid"]) if h and h["gid"] is not None else "-",
               f"an IPv4 address (RoCE v2 GID) on {rd[i]['name']}'s {hca or 'IB_HCA'}")
        if len(rd) == 3:
            it("engine", None, "fail", "3 nodes", "an engine that splits GLM-5.3 over three ranks (TP=3): not in this release")
    return {"mode": mode, "items": items}


def text(res):
    """A few lines for a person."""
    L = []
    mode = res["mode"] or "none"
    L.append(f"topology: {res['verdict']}" + (f" -> mode {mode}" if res["mode"] else "") + f" (TOPOLOGY={res['conf']}"
             + ("" if res["agrees"] or res["conf"] == "auto" else ", does not agree with the cabling") + ")"
             + (f": {res['why']}" if res["why"] else ""))
    L.append("  order: " + " ".join(f"{n}({a})" for n, a in zip(res["names"], res["order"])) + ("  (NODES reordered)" if res["reorder"] else ""))
    names = {n["rank"]: n["name"] or n["addr"] for n in res["nodes"]}
    for lk in res["links"]:
        L.append(f"  link {names[lk['a']]} {lk['pa']} - {names[lk['b']]} {lk['pb']}: lanes {''.join('+' if x else '-' for x in lk['lanes'])}"
                 f", {' / '.join(s or '-' for s in lk['subnets'])}{'' if lk['crossed'] else ', not crossed'}{', seen from one side' if lk['one_side'] else ''}"
                 f"{' (by addresses, no LLDP)' if lk['via'] == 'addresses' else ''}")
    if res["fabric"]:
        f = res["fabric"]
        L.append(f"  switch {' '.join(f['switches']) or '?'} on {f['subnet'] or '-'}" + (" (split)" if f["split"] else ""))
        for m in f["members"]:
            L.append(f"    {names[m['rank']]}: {m['dev']} {m['addr']} at {m['port']}")
    for p in res["problems"]:
        L.append(f"  problem {p['what']}: {p['detail']} -> {p['fix']}")
    for x in res["prereq"]["items"]:
        if x["state"] != "ok":
            L.append(f"  needs {x['key']}" + (f" on {names[x['rank']]}" if x["rank"] is not None else "") + f": {x['value']} -> {x['fix']}")
    if res["multi"]["why"]:
        L.append(f"  MULTI: up to {res['multi']['max_lane']} ({res['multi']['why']})")
    return "\n".join(L)


def shell(res):
    """Lines tf-glm53 evaluates: the mode, NODES in its order, and the devices the mode takes."""
    q = lambda s: "'" + str(s).replace("'", "'\\''") + "'"  # noqa: E731
    return "\n".join([f"TOPO_MODE={q(res['mode'] or '')}", f"TOPO_VERDICT={q(res['verdict'])}", f"TOPO_AGREES={q(int(res['agrees']))}",
                      "TOPO_NODES=(" + " ".join(q(x) for x in res["order"]) + ")", f"TOPO_HCA={q(res['hca'])}",
                      f"TOPO_IFNAME={q(res['ifname'])}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nodes", required=True, help="NODES, rank 0 first, separated by spaces")
    ap.add_argument("--conf", default="auto", help="glm53.conf's TOPOLOGY: auto, full or ring4")
    ap.add_argument("--hca", default="", help="glm53.conf's IB_HCA (a switch: the half preferred)")
    ap.add_argument("--runtime", default="host")
    ap.add_argument("--dcp", type=int, default=2)
    ap.add_argument("--nccl-lib", default="")
    ap.add_argument("--nccl-sha256", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--shell", action="store_true", help="the lines tf-glm53 evaluates (the text goes to stderr)")
    ap.add_argument("--save", default="", help="also write the JSON to this file")
    ap.add_argument("--dump", default="", help="also write what the nodes said (for --readings)")
    ap.add_argument("--readings", default="", help="judge what --dump kept instead of reading the nodes")
    a = ap.parse_args()
    a.nodes = a.nodes.split()
    res = judge(a)
    if a.save:
        with open(a.save, "w") as f:
            json.dump(res, f, indent=1)
    if a.shell:
        print(text(res), file=sys.stderr)
    print(json.dumps(res, indent=1) if a.json else shell(res) if a.shell else text(res))
    return 0 if res["mode"] and res["agrees"] else 3


if __name__ == "__main__":
    sys.exit(main())
