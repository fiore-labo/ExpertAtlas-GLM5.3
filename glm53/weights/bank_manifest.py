#!/usr/bin/env python3
"""bank_manifest.py: the manifest of an Expert Atlas bank, and the bank list (expert-plan/banks.json) that names it.

A bank is a directory of experts-layer-NNN.safetensors files (routed experts of layers 3..77, stored TP-sliced:
model.layers.L.mlp.experts.E.{gate,up,down}_proj.rankR.{trellis,suh,svh,mcg|mul1}). The manifest records, per layer
file: size, sha256 of the file, sha256 of the raw header bytes, the trellis widths (K2 = 2 x bits) per projection, the
codebook, and the xxh3-64 of every tensor's bytes in data-offset order (one tensor = one stored rank's share). It also
checks that every expert, projection and stored rank has the declared bits and the same codebook, and that the file
sha256 equals the bank's SHA256SUMS line when there is one.

  make:  bank_manifest.py make BANK_DIR --id ID --bits N --calibration TEXT [--out FILE] [--jobs 6] [--layers 3-77]
         (FILE defaults to BANK_DIR/ID.json, where the launcher and the engine look for it)
  check: bank_manifest.py check BANK_DIR MANIFEST    headers and sizes only (cheap); exit 1 on any difference
  banks: bank_manifest.py banks OUT BANK_DIR...      expert-plan/banks.json for these banks (each with its ID.json):
         the engine loads a bank only when its manifest's sha256 is the one banks.json gives

Files are read with O_DIRECT (no page cache: GB10 shares it with the GPU). xxh3-64 is XXH3_64bits with seed 0
(python xxhash.xxh3_64), the same function as Zig's std.hash.XxHash3.hash(0, bytes). Needs xxhash (pip install xxhash).
"""
import argparse, base64, hashlib, json, mmap, os, re, struct, sys, time
from concurrent.futures import ProcessPoolExecutor

FORMAT, VERSION = "glm53-expert-bank", 1
NAME = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.rank(\d+)\.(trellis|suh|svh|mcg|mul1)$")
CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return n, raw, json.loads(raw)


def layout(layer, h, bits):
    """Check one layer's header; return (k2 per projection, codebook, experts, ranks, tensors in data order)."""
    meta = h.pop("__metadata__", None)
    seen, k2s, cbs, problems = {}, {"gate": set(), "up": set(), "down": set()}, set(), []
    for k, v in h.items():
        m = NAME.match(k)
        if not m:
            problems.append(f"unexpected tensor {k}")
            continue
        L, e, proj, r, part = int(m[1]), int(m[2]), m[3], int(m[4]), m[5]
        if L != layer:
            problems.append(f"{k}: layer {L} in the file of layer {layer}")
        seen.setdefault((e, proj, r), {})[part] = v
    experts = 1 + max(e for e, _, _ in seen)
    ranks = 1 + max(r for _, _, r in seen)
    for e in range(experts):
        for proj in ("gate", "up", "down"):
            for r in range(ranks):
                t = seen.get((e, proj, r))
                if t is None:
                    problems.append(f"expert {e} {proj} rank {r} missing")
                    continue
                tr = t.get("trellis")
                if tr is None or tr["dtype"] != "I16" or len(tr["shape"]) != 3 or tr["shape"][2] % 8:
                    problems.append(f"expert {e} {proj} rank {r}: bad trellis {tr and (tr['dtype'], tr['shape'])}")
                    continue
                k2 = tr["shape"][2] // 8
                k2s[proj].add(k2)
                if k2 != 2 * bits:
                    problems.append(f"expert {e} {proj} rank {r}: K2 {k2} is not {2 * bits} ({bits} bits)")
                for s in ("suh", "svh"):
                    if s not in t or t[s]["dtype"] != "F16" or len(t[s]["shape"]) != 1:
                        problems.append(f"expert {e} {proj} rank {r}: bad {s}")
                cb = [c for c in ("mcg", "mul1") if c in t]
                if len(cb) != 1:
                    problems.append(f"expert {e} {proj} rank {r}: codebook markers {cb}")
                else:
                    cbs.add(cb[0])
    if len(cbs) > 1:
        problems.append(f"mixed codebooks {sorted(cbs)}")
    order = sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0])
    pos = 0
    for k, v in order:
        a, b = v["data_offsets"]
        if a != pos:
            problems.append(f"{k}: data starts at {a}, expected {pos} (gap or overlap)")
        pos = b
    return {p: sorted(s) for p, s in k2s.items()}, sorted(cbs), experts, ranks, order, pos, meta, problems


def hash_file(path, data_start, order, data_len):
    """One O_DIRECT pass: sha256 of the whole file and xxh3-64 of every tensor (data-offset order)."""
    import xxhash
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, CHUNK)
    mv = memoryview(buf)
    whole = hashlib.sha256()
    hashes = bytearray()
    bounds = [(data_start + v["data_offsets"][0], data_start + v["data_offsets"][1]) for _, v in order]
    i, cur = 0, xxhash.xxh3_64()
    off = 0
    try:
        while True:
            n = os.preadv(fd, [buf], off)
            if n <= 0:
                break
            with mv[:n] as chunk:
                whole.update(chunk)
                # feed the tensors that overlap [off, off + n)
                while i < len(bounds):
                    a, b = bounds[i]
                    lo, hi = max(a, off), min(b, off + n)
                    if lo < hi:
                        with chunk[lo - off:hi - off] as part:
                            cur.update(part)
                    if b <= off + n:
                        hashes += cur.intdigest().to_bytes(8, "little")
                        cur = xxhash.xxh3_64()
                        i += 1
                    else:
                        break
            off += n
            if n < CHUNK:
                break
    finally:
        os.close(fd)
        mv.release()
        buf.close()
    if i != len(bounds):
        raise RuntimeError(f"{path}: file ended inside tensor {i} of {len(bounds)}")
    if off != data_start + data_len:
        raise RuntimeError(f"{path}: {off} bytes read, header says {data_start + data_len}")
    return whole.hexdigest(), base64.b64encode(bytes(hashes)).decode()


def one_layer(args):
    bank, layer, bits, sums = args
    f = f"experts-layer-{layer:03d}.safetensors"
    path = os.path.join(bank, f)
    n, raw, h = read_header(path)
    k2, cb, experts, ranks, order, data_len, meta, problems = layout(layer, h, bits)
    size = os.path.getsize(path)
    if size != 8 + n + data_len:
        problems.append(f"file size {size} != 8 + header {n} + data {data_len}")
    if problems:
        return {"layer": layer, "file": f, "problems": problems[:20], "problem_count": len(problems)}
    sha, shares = hash_file(path, 8 + n, order, data_len)
    if sums is not None and sums.get(f) not in (None, sha):
        problems.append(f"sha256 {sha} differs from SHA256SUMS {sums[f]}")
    out = {"layer": layer, "file": f, "size": size, "sha256": sha, "header_bytes": n,
           "header_sha256": hashlib.sha256(raw).hexdigest(), "k2": k2, "codebook": cb[0], "experts": experts,
           "stored_ranks": ranks, "tensors": len(order), "shares_xxh3": shares}
    if sums is not None and f not in sums:
        out["note"] = "not in SHA256SUMS"
    if problems:
        out["problems"] = problems
    return out


def read_sums(bank):
    p = os.path.join(bank, "SHA256SUMS")
    if not os.path.exists(p):
        return None
    sums = {}
    for line in open(p):
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip().lstrip("*")] = parts[0]
    return sums


def layer_range(s):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


def make(a):
    sums = read_sums(a.bank)
    t0 = time.time()
    jobs = [(a.bank, L, a.bits, sums) for L in layer_range(a.layers)]
    with ProcessPoolExecutor(a.jobs) as ex:
        layers = list(ex.map(one_layer, jobs))
    bad = [x for x in layers if "problems" in x]
    for x in bad:
        print(f"layer {x['layer']}: {x.get('problem_count', len(x['problems']))} problems: {x['problems'][:3]}", file=sys.stderr)
    if bad:
        sys.exit(1)
    m = {"format": FORMAT, "version": VERSION, "id": a.id, "bits": a.bits, "calibration": a.calibration,
         "base_model": a.base_model, "quantization": "exl3 trellis", "codebook": layers[0]["codebook"],
         "stored_ranks": layers[0]["stored_ranks"], "experts": layers[0]["experts"],
         "made_unix": int(time.time()), "first_layer": layers[0]["layer"], "layer_count": len(layers), "layers": layers}
    text = json.dumps(m, indent=1, sort_keys=True) + "\n"
    a.out = a.out or os.path.join(a.bank, a.id + ".json")
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, a.out)
    gib = sum(x["size"] for x in layers) / 2**30
    print(f"MANIFEST {a.out}: {a.id}, {a.bits} bits, {len(layers)} layers, {gib:.1f} GiB in {time.time() - t0:.0f} s, "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def check(a):
    m = json.load(open(a.manifest))
    if m.get("format") != FORMAT or m.get("version") != VERSION:
        sys.exit(f"{a.manifest}: not a {FORMAT} v{VERSION} manifest")
    bad = 0
    for x in m["layers"]:
        path = os.path.join(a.bank, x["file"])
        try:
            n, raw, _ = read_header(path)
            size = os.path.getsize(path)
        except OSError as e:
            print(f"{x['file']}: {e}"); bad += 1; continue
        if size != x["size"] or n != x["header_bytes"] or hashlib.sha256(raw).hexdigest() != x["header_sha256"]:
            print(f"{x['file']}: size/header differs from the manifest"); bad += 1
    sums = read_sums(a.bank)
    if sums is not None:
        for x in m["layers"]:
            if sums.get(x["file"]) not in (None, x["sha256"]):
                print(f"{x['file']}: SHA256SUMS differs from the manifest"); bad += 1
    if bad:
        print(f"CHECK FAILED {a.bank}: {bad} of {len(m['layers'])} layer files differ from {m['id']}"); sys.exit(1)
    print(f"CHECK OK {a.bank}: {len(m['layers'])} layer files = {m['id']} ({m['bits']} bits)")


def banks(a):
    out = {"format": "glm53-expert-banks", "version": 1,
           "note": "the routed experts of layers 3..77, one directory a width on every node, each with its manifest <id>.json",
           "banks": []}
    for d in a.dirs:
        found = [f for f in os.listdir(d) if f.endswith(".json")]
        ms = [json.load(open(os.path.join(d, f))) for f in found]
        ms = [(f, m) for f, m in zip(found, ms) if m.get("format") == FORMAT and f == m["id"] + ".json"]
        if len(ms) != 1:
            sys.exit(f"{d}: {len(ms)} bank manifests named <id>.json (need one)")
        f, m = ms[0]
        out["banks"].append({"id": m["id"], "bits": m["bits"], "calibration": m["calibration"],
                             "dir": os.path.abspath(d).replace(os.path.expanduser("~"), "~", 1),
                             "manifest_sha256": hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest(),
                             "layers": m["layer_count"], "first_layer": m["first_layer"],
                             "bytes": sum(x["size"] for x in m["layers"]), "codebook": m["codebook"],
                             "stored_ranks": m["stored_ranks"]})
    out["banks"].sort(key=lambda b: b["bits"])
    if len({b["bits"] for b in out["banks"]}) != len(out["banks"]):
        sys.exit("two banks of the same width")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for b in out["banks"]:
        print(f"{b['bits']} bits: {b['id']} in {b['dir']}, manifest {b['manifest_sha256'][:16]}")
    print(f"BANKS {a.out}: {len(out['banks'])} banks")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("bank"); mk.add_argument("--id", required=True); mk.add_argument("--bits", type=int, required=True)
    mk.add_argument("--calibration", required=True); mk.add_argument("--out")
    mk.add_argument("--base-model", default="zai-org/GLM-5.3-BF16@9d2398f")
    mk.add_argument("--layers", default="3-77"); mk.add_argument("--jobs", type=int, default=6)
    ck = sub.add_parser("check")
    ck.add_argument("bank"); ck.add_argument("manifest")
    bk = sub.add_parser("banks")
    bk.add_argument("out"); bk.add_argument("dirs", nargs="+")
    a = p.parse_args()
    {"make": make, "check": check, "banks": banks}[a.cmd](a)


if __name__ == "__main__":
    main()
