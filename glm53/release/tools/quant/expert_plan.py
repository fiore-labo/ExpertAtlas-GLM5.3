#!/usr/bin/env python3
"""expert_plan.py: expert plans for the TensorFold GLM-5.3 engine.

A plan says, for every routed expert of layers 3..77, which bank it is loaded from at start (TF_GLM53_EXPERT_PLAN).
Banks are listed in expert-plan/banks.json (id, bits, dir, manifest sha256, and the manifest's path when it is not
<id>.json in the bank's directory).

  uniform  ID --out PLAN                          every expert from one bank
  from-set SET --map 3:ID,4:ID --out PLAN         a saved set (glm53-expert-bits v1) with a bank per width
  mix      --bits FILE --map 2:ID,... --out PLAN  bits from a raw file (one byte an expert, layer-major)
  tiers    --run DIR BASE DOM... [--run ... --weight W] --shares 2:0.25,3:0.25,4:0.25,5:0.25 --map 2:ID,... --out PLAN
                                                  bits by usage: experts ordered by importance (the largest weighted
                                                  share of its layer's picks over the domains), all
                                                  layers together; the least important share gets the fewest bits
  optimize --run ... [--weight W] --delta 2:D2,3:D3,4:0,5:D5 --mean-bits B --map ... --out PLAN
                                                  bits that minimise sum_e w_e * delta(bits_e) under a byte budget (the
                                                  mean bits B): w_e = the expert's weighted mean share of its layer's
                                                  picks over the domains / layers; delta = the measured loss of the
                                                  whole model at that width (e.g. KL - KL at 4 bits); Lagrangian
  show     PLAN [--tp 4]                          counts, mean bits and the routed experts' device bytes per rank
  digest   PLAN                                   the plan digest (sha256 of the file), as the engine computes it

Plan format glm53-expert-plan v1 (JSON):
  {"format", "version", "name", "first_layer", "layers", "experts",
   "banks": [{"id", "bits", "manifest_sha256", "dir", ["manifest"]}, ...],   bank_of indexes this list
   "bank_of": base64, one byte an expert (layer-major from first_layer),
   "counts": {"bits": n}, "mean_bits", "made_unix", "note"}
The engine reads banks[].dir unless TF_GLM53_PLAN_BANK_DIRS=ID:DIR,... overrides it on a node.
"""
import argparse, base64, hashlib, json, os, sys, time

FORMAT, VERSION = "glm53-expert-plan", 1
HERE = os.path.dirname(os.path.abspath(__file__))
BANKS = os.path.normpath(os.path.join(HERE, "..", "..", "expert-plan", "banks.json"))
# GLM-5.3: hidden D, routed expert intermediate I, experts a layer, first and last MoE layers with routed experts
D, I, E, FIRST, LAYERS = 6144, 2048, 256, 3, 75


def load_banks(path):
    """banks.json by id; a leading ~ in a dir or manifest is the home directory (every node has the same)."""
    b = json.load(open(path))
    assert b["format"] == "glm53-expert-banks" and b["version"] == 1, path
    for x in b["banks"]:
        for k in ("dir", "manifest"):
            if x.get(k):
                x[k] = os.path.expanduser(x[k])
    return {x["id"]: x for x in b["banks"]}


def parse_map(s, banks):
    out = {}
    for item in s.split(","):
        bits, bid = item.split(":", 1)
        if bid not in banks:
            sys.exit(f"bank {bid} is not in {BANKS}")
        if banks[bid]["bits"] != int(bits):
            sys.exit(f"bank {bid} has {banks[bid]['bits']} bits, not {bits}")
        out[int(bits)] = bid
    return out


def write_plan(out, name, bank_ids, bank_of, banks, note=""):
    if len(bank_of) != LAYERS * E:
        sys.exit(f"{len(bank_of)} experts, want {LAYERS * E}")
    used = sorted(set(bank_of))
    if used[-1] >= len(bank_ids):
        sys.exit("bank index out of range")
    bits = [banks[bank_ids[i]]["bits"] for i in bank_of]
    counts = {}
    for b in bits:
        counts[str(b)] = counts.get(str(b), 0) + 1
    plan = {"format": FORMAT, "version": VERSION, "name": name, "first_layer": FIRST, "layers": LAYERS, "experts": E,
            "banks": [dict({"id": i, "bits": banks[i]["bits"], "manifest_sha256": banks[i]["manifest_sha256"], "dir": banks[i]["dir"]},
                           **({"manifest": banks[i]["manifest"]} if banks[i].get("manifest") else {})) for i in bank_ids],
            "bank_of": base64.b64encode(bytes(bank_of)).decode(), "counts": dict(sorted(counts.items())),
            "mean_bits": round(sum(bits) / len(bits), 4), "made_unix": int(time.time()), "note": note}
    text = json.dumps(plan, indent=1) + "\n"
    with open(out + ".tmp", "w") as f:
        f.write(text)
    os.replace(out + ".tmp", out)
    print(f"PLAN {out}: {name}, mean {plan['mean_bits']} bits, counts {plan['counts']}, digest {hashlib.sha256(text.encode()).hexdigest()}")


def read_plan(path):
    p = json.load(open(path))
    if p.get("format") != FORMAT or p.get("version") != VERSION:
        sys.exit(f"{path}: not a {FORMAT} v{VERSION}")
    return p


def device_bytes(bits, tp):
    """One routed expert's bytes on one rank: trellis 3*D*I/tp/8*bits (gate/up/down), fp16 scales: the D-sized ones
    (gate/up suh, down svh) replicated, the I-sized ones split (split.zig), slab alignment ignored (256 B)."""
    return 3 * D * (I // tp) // 8 * bits + 2 * 3 * (D + I // tp)


def cmd_uniform(a, banks):
    write_plan(a.out, a.name or f"all {banks[a.bank]['bits']} bit ({a.bank})", [a.bank], [0] * (LAYERS * E), banks, a.note)


def to_plan(bitsb, m, name, out, banks, note):
    ids = [m[b] for b in sorted(m)]
    idx = {b: i for i, b in enumerate(sorted(m))}
    missing = sorted(set(bitsb) - set(m))
    if missing:
        sys.exit(f"no bank for widths {missing}: give --map")
    write_plan(out, name, ids, [idx[b] for b in bitsb], banks, note)


def cmd_from_set(a, banks):
    s = json.load(open(a.set))
    if s.get("format") != "glm53-expert-bits" or s.get("version") != 1:
        sys.exit(f"{a.set}: not a glm53-expert-bits v1 set")
    if (s["first_layer"], s["layers"], s["experts"]) != (FIRST, LAYERS, E):
        sys.exit(f"{a.set}: shape {(s['first_layer'], s['layers'], s['experts'])}")
    to_plan(base64.b64decode(s["bits"]), parse_map(a.map, banks), a.name or s.get("name", os.path.basename(a.set)), a.out, banks,
            a.note or f"from set {os.path.basename(a.set)}")


def cmd_mix(a, banks):
    to_plan(open(a.bits, "rb").read(), parse_map(a.map, banks), a.name or os.path.basename(a.bits), a.out, banks, a.note)


def load_stats(path):
    """glm53-expert-stats v1: header line, then u64 counts [2][L][E] (prompt, decode) and fp32 weight sums."""
    import struct
    data = open(path, "rb").read(); nl = data.index(b"\n"); h = data[:nl].decode().split()
    L, E_ = int(h[3]), int(h[5])
    return L, E_, struct.unpack_from(f"<{2 * L * E_}Q", data, nl + 1)


def summed_stats(d, name, ranks=8):
    tot = None
    for r in range(ranks):
        L, E_, c = load_stats(f"{d}/{name}.rank{r}")
        tot = list(c) if tot is None else [x + y for x, y in zip(tot, c)]
    return L, E_, tot


def importance(runs, weights):
    """Each domain's use (its snapshot minus the one before, all ranks, prompt and decode) as each expert's share of its
    layer's picks; importance = the largest weighted share over the domains."""
    share, wt = {}, {}
    for (d, base, *doms), w in zip(runs, weights):
        snaps = [summed_stats(d, n) for n in [base] + doms]
        L = snaps[0][0]
        for i, dom in enumerate(doms):
            prev, cur = snaps[i][2], snaps[i + 1][2]
            share[dom] = {}; wt[dom] = w
            for l in range(FIRST, FIRST + LAYERS):
                row = [sum(cur[(p * L + l) * E + e] - prev[(p * L + l) * E + e] for p in (0, 1)) for e in range(E)]
                t = sum(row); share[dom][l] = [x / t for x in row]
    return {(l, e): max(wt[d] * share[d][l][e] for d in share) for l in range(FIRST, FIRST + LAYERS) for e in range(E)}


def shares_by_domain(runs, weights):
    """{domain: {layer: [share of the layer's picks per expert]}} and the domains' weights (as importance())."""
    share, wt = {}, {}
    for (d, base, *doms), w in zip(runs, weights):
        snaps = [summed_stats(d, n) for n in [base] + doms]
        L = snaps[0][0]
        for i, dom in enumerate(doms):
            prev, cur = snaps[i][2], snaps[i + 1][2]
            share[dom] = {}; wt[dom] = w
            for l in range(FIRST, FIRST + LAYERS):
                row = [sum(cur[(p * L + l) * E + e] - prev[(p * L + l) * E + e] for p in (0, 1)) for e in range(E)]
                t = sum(row); share[dom][l] = [x / t for x in row]
    return share, wt


def cmd_optimize(a, banks):
    weights = [1.0] + (a.weight + [1.0] * len(a.run))[:len(a.run) - 1]
    share, wt = shares_by_domain(a.run, weights)
    tw = sum(wt.values())
    # w_e: weighted mean share over the domains, over the layers (sums to 1 over all experts)
    w = {(l, e): sum(wt[d] * share[d][l][e] for d in share) / tw / LAYERS for l in range(FIRST, FIRST + LAYERS) for e in range(E)}
    delta = {int(b): float(x) for b, x in (t.split(":") for t in a.delta.split(","))}
    widths = sorted(delta)
    cost = {b: device_bytes(b, a.tp) for b in widths}
    target = LAYERS * E * (3 * D * (I // a.tp) // 8 * a.mean_bits + 2 * 3 * (D + I // a.tp))

    def pick(lam):
        out = {k: min(widths, key=lambda b: (w[k] * delta[b] + lam * cost[b], b)) for k in w}
        return out, sum(cost[b] for b in out.values())

    lo, hi = 0.0, 1.0
    while pick(hi)[1] > target:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if pick(mid)[1] > target:
            lo = mid
        else:
            hi = mid
    bits, used = pick(hi)
    pred = sum(w[k] * delta[b] for k, b in bits.items())
    flat = bytes(bits[(l, e)] for l in range(FIRST, FIRST + LAYERS) for e in range(E))
    print(f"optimize: lambda {hi:.3e}, {used / 2**30:.2f} GiB a rank at TP{a.tp} (target {target / 2**30:.2f}), predicted loss {pred:.5f}")
    note = a.note or (f"optimize mean {a.mean_bits} bits, delta {a.delta}, over " + ", ".join(" ".join(r[2:]) for r in a.run))
    to_plan(flat, parse_map(a.map, banks), a.name or f"optimize {a.mean_bits} bits", a.out, banks, note)


def cmd_tiers(a, banks):
    weights = [1.0] + (a.weight + [1.0] * len(a.run))[:len(a.run) - 1]
    imp = importance(a.run, weights)
    order = sorted(imp, key=lambda k: imp[k])
    shares = sorted((int(b), float(f)) for b, f in (x.split(":") for x in a.shares.split(",")))
    if abs(sum(f for _, f in shares) - 1) > 1e-9:
        sys.exit(f"shares add up to {sum(f for _, f in shares)}, not 1")
    bits = {}
    n, cum, start = len(order), 0.0, 0
    for b, f in shares:
        cum += f
        end = n if b == shares[-1][0] else int(round(cum * n))
        for k in order[start:end]:
            bits[k] = b
        start = end
    flat = bytes(bits[(l, e)] for l in range(FIRST, FIRST + LAYERS) for e in range(E))
    note = a.note or ("tiers " + a.shares + " by importance over " + ", ".join(" ".join(r[2:]) for r in a.run)
                      + (f" (weights {weights})" if len(a.run) > 1 else ""))
    to_plan(flat, parse_map(a.map, banks), a.name or ("tiers " + a.shares), a.out, banks, note)


def cmd_show(a, banks):
    p = read_plan(a.plan)
    of = base64.b64decode(p["bank_of"])
    bits = [p["banks"][i]["bits"] for i in of]
    tot = sum(device_bytes(b, a.tp) for b in bits) + 36 * len(bits)
    print(f"{a.plan}: {p['name']}; mean {sum(bits) / len(bits):.4f} bits, counts {p['counts']}; banks "
          + ", ".join(f"{b['id']} ({b['bits']} bit)" for b in p["banks"]))
    print(f"routed experts on one rank at TP{a.tp}: {tot / 2**30:.3f} GiB (trellis + scales + tables, without slab alignment)")
    per = [0] * LAYERS
    for j, b in enumerate(bits):
        per[j // E] += b
    print("mean bits by layer: " + " ".join(f"{x / E:.2f}" for x in per))


def cmd_digest(a, banks):
    print(hashlib.sha256(open(a.plan, "rb").read()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--banks", default=BANKS)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("uniform"); u.add_argument("bank"); u.add_argument("--out", required=True)
    f = sub.add_parser("from-set"); f.add_argument("set"); f.add_argument("--map", required=True); f.add_argument("--out", required=True)
    m = sub.add_parser("mix"); m.add_argument("--bits", required=True); m.add_argument("--map", required=True); m.add_argument("--out", required=True)
    for x in (u, f, m):
        x.add_argument("--name"); x.add_argument("--note", default="")
    t = sub.add_parser("tiers"); t.add_argument("--run", nargs="+", action="append", required=True)
    t.add_argument("--weight", type=float, action="append", default=[]); t.add_argument("--shares", required=True)
    t.add_argument("--map", required=True); t.add_argument("--out", required=True)
    t.add_argument("--name"); t.add_argument("--note", default="")
    o = sub.add_parser("optimize"); o.add_argument("--run", nargs="+", action="append", required=True)
    o.add_argument("--weight", type=float, action="append", default=[]); o.add_argument("--delta", required=True)
    o.add_argument("--mean-bits", type=float, required=True); o.add_argument("--tp", type=int, default=4)
    o.add_argument("--map", required=True); o.add_argument("--out", required=True)
    o.add_argument("--name"); o.add_argument("--note", default="")
    s = sub.add_parser("show"); s.add_argument("plan"); s.add_argument("--tp", type=int, default=4)
    d = sub.add_parser("digest"); d.add_argument("plan")
    a = p.parse_args()
    banks = load_banks(a.banks)
    {"uniform": cmd_uniform, "from-set": cmd_from_set, "mix": cmd_mix, "tiers": cmd_tiers, "optimize": cmd_optimize, "show": cmd_show, "digest": cmd_digest}[a.cmd](a, banks)


if __name__ == "__main__":
    main()
