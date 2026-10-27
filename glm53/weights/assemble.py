#!/usr/bin/env python3
"""assemble.py: this recipe's weight files from your own exllamav3 conversions of zai-org/GLM-5.3-BF16 (README.md).

  bank QTENSORS OUT [--layers 3-77] [--jobs 2]
      the routed experts of one conversion (its work directory's qtensors/) as an Expert Atlas bank: one file a layer,
      OUT/experts-layer-NNN.safetensors, each expert's matrices cut into the eight tensor-parallel shares the engine
      loads (whole tiles; nothing is quantized again)
  checkpoint --official BF16 --experts BANK4 --mtp-qtensors Q4 --nonexperts QNE --mtp-linears FILE --out OUT [--copy]
      the eight-node checkpoint: the 4-bit bank's expert files, the non-expert linears, the MTP layer and the tensors
      kept unquantized (copied from the official files as stored), the official small files, config.json with a
      quantization_config, the tensor index and SHA256SUMS
  mix --base CKPT --bank3 BANK3 --selection q3-selection.json --out OUT [--copy]
      the four-node checkpoint: CKPT with the experts the selection lists ({"layer": [expert, ...]}) at 3 bits
  check DIR
      a checkpoint's index against its files, and the widths and codebooks of its quantized linears

Expert and checkpoint files from different runs are the same bytes when their inputs are: the tensors are laid out as
safetensors' own writer lays them out and the metadata is written in a fixed order. Python 3.9+ and numpy (bank only).
Files are written next to their final name and renamed when complete; a file already there is kept (so a run can be
started again), except SHA256SUMS, the index and config.json, which are always written again.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REV = "9d2398f"                      # zai-org/GLM-5.3-BF16 at 9d2398f478cab2de883137db3a36ad2c96205e24
FIRST, LAST, MTP = 3, 77, 78         # layers with routed experts; the MTP layer
EXPERTS, WORLD = 256, 8
PIECE = 64 << 20
ROUTED = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
# safetensors' dtype enum in its order: its writer lays tensors out by dtype (descending), then by name
DTYPES = ["BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "I16", "U16", "F16", "BF16",
          "I32", "U32", "F32", "F64", "I64", "U64"]
OFFICIAL_SMALL = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mem_guard(floor_gib=20):
    """Stop when the host runs low (a DGX Spark's GPU shares this memory)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                gib = int(line.split()[1]) / 2**20
                if gib < floor_gib:
                    sys.exit(f"MemAvailable {gib:.1f} GiB < {floor_gib}: stopped")
                return
    except FileNotFoundError:
        pass


def drop_cache(fd):
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


class Tensors:
    """A safetensors file: its header and its tensors' bytes."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        h = json.loads(self.f.read(n))
        self.meta = h.pop("__metadata__", None)
        self.h = h
        self.start = 8 + n

    def nbytes(self, k):
        s, e = self.h[k]["data_offsets"]
        return e - s

    def read(self, k):
        s, e = self.h[k]["data_offsets"]
        self.f.seek(self.start + s)
        b = self.f.read(e - s)
        if len(b) != e - s:
            sys.exit(f"{self.path}: {k} cut short")
        return b

    def pieces(self, k):
        s, e = self.h[k]["data_offsets"]
        self.f.seek(self.start + s)
        p = s
        while p < e:
            b = self.f.read(min(PIECE, e - p))
            if not b:
                sys.exit(f"{self.path}: {k} cut short")
            p += len(b)
            yield b

    def data_order(self):
        return sorted(self.h, key=lambda k: self.h[k]["data_offsets"][0])

    def close(self):
        drop_cache(self.f.fileno())
        self.f.close()


class Out:
    """One tensor to write: from a source file's tensor, or bytes made here."""

    def __init__(self, name, dtype, shape, nbytes, src=None, key=None, make=None):
        self.name, self.dtype, self.shape, self.nbytes = name, dtype, list(shape), nbytes
        self.src, self.key, self.make = src, key or name, make

    @classmethod
    def copy(cls, src, key, name=None):
        v = src.h[key]
        return cls(name or key, v["dtype"], v["shape"], src.nbytes(key), src=src, key=key)

    def pieces(self):
        if self.make is not None:
            yield self.make()
        else:
            yield from self.src.pieces(self.key)


def st_order(outs):
    return sorted(outs, key=lambda t: (-DTYPES.index(t.dtype), t.name))


def write_tensors(path, outs, meta):
    """outs in the order given (data and header alike); meta: the __metadata__ entries in the order given."""
    h, off = {"__metadata__": meta}, 0
    for t in outs:
        h[t.name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.nbytes]}
        off += t.nbytes
    raw = json.dumps(h, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for t in outs:
            n = 0
            for b in t.pieces():
                f.write(b)
                n += len(b)
            if n != t.nbytes:
                sys.exit(f"{path}: {t.name} wrote {n} bytes, expected {t.nbytes}")
        f.flush()
        os.fsync(f.fileno())
        drop_cache(f.fileno())
    os.replace(tmp, path)
    return off


# ---- bank -------------------------------------------------------------------------------------------------------

NP = {"I16": "int16", "F16": "float16", "BF16": "uint16", "I32": "int32", "U32": "uint32", "F32": "float32",
      "U16": "uint16", "U8": "uint8", "I8": "int8"}


def count(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def reslice_layer(job):
    """One layer of a conversion's qtensors -> experts-layer-NNN.safetensors in tensor-parallel shares:
    gate_proj / up_proj (the output N split): trellis[:, r*N/16/W ...], suh whole, svh[r*N/W ...]
    down_proj (the input K split): trellis[r*K/16/W ...], suh[r*K/W ...], svh whole; the codebook marker on every rank."""
    import numpy as np
    qdir, layer, out_dir = job
    dst = os.path.join(out_dir, f"experts-layer-{layer:03d}.safetensors")
    if os.path.exists(dst):
        return f"layer {layer}: already there"
    mem_guard()
    src = Tensors(os.path.join(qdir, f"model.layers.{layer}.safetensors"))
    cache = {}

    def arr(key):
        if key not in cache:
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            v = src.h[key]
            cache[key] = np.frombuffer(src.read(key), dtype=NP[v["dtype"]]).reshape(v["shape"])
        return cache[key]

    def piece(p, part, r):
        tr = src.h[p + ".trellis"]["shape"]
        kt, nt = tr[0], tr[1]
        if part == "trellis":
            a = arr(p + ".trellis")
            if p.endswith("down_proj"):
                s = kt // WORLD
                return a[r * s:(r + 1) * s]
            s = nt // WORLD
            return a[:, r * s:(r + 1) * s]
        a = arr(p + "." + part)
        if (part == "suh") == p.endswith("down_proj"):  # the split side: suh of down_proj, svh of gate / up
            s = (kt if part == "suh" else nt) * 16 // WORLD
            return a[r * s:(r + 1) * s]
        return a

    outs = []
    for e in range(EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            p = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
            marks = [m for m in ("mcg", "mul1") if p + "." + m in src.h]
            if p + ".trellis" not in src.h or len(marks) != 1:
                sys.exit(f"{src.path}: {p} is not one EXL3 linear (trellis + one codebook marker)")
            tr = src.h[p + ".trellis"]["shape"]
            kt, nt = tr[0], tr[1]
            split = kt if proj == "down_proj" else nt
            if split % WORLD:
                sys.exit(f"{p}: {split} tiles do not split into {WORLD} shares")
            for r in range(WORLD):
                q = f"{p}.rank{r}"
                for part in ("trellis", "suh", "svh"):
                    v = src.h[p + "." + part]
                    shape = list(v["shape"])
                    if part == "trellis":
                        shape[0 if proj == "down_proj" else 1] //= WORLD
                    elif (part == "suh") == (proj == "down_proj"):
                        shape[0] //= WORLD
                    size = src.nbytes(p + "." + part) // count(v["shape"]) * count(shape)
                    outs.append(Out(f"{q}.{part}", v["dtype"], shape, size,
                                    make=lambda p=p, part=part, r=r: np.ascontiguousarray(piece(p, part, r)).tobytes()))
                outs.append(Out.copy(src, p + "." + marks[0], f"{q}.{marks[0]}"))
    write_tensors(dst, st_order(outs), {"format": "pt", "reshard": "exllamav3 whole matrices -> tp8 (pure tile re-slice, no requantization)"})
    src.close()
    return f"layer {layer}: {len(outs)} tensors"


def cmd_bank(a):
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (int(x) for x in a.layers.split("-"))
    jobs = [(a.qtensors, L, a.out) for L in range(lo, hi + 1)]
    with ProcessPoolExecutor(a.jobs) as ex:
        for msg in ex.map(reslice_layer, jobs):
            log(msg)
    log(f"BANK {a.out}: layers {lo}-{hi}")


# ---- checkpoint -------------------------------------------------------------------------------------------------

def stem(k):
    return k.rsplit(".", 1)[0]


def quantized_stems(t, routed):
    """The EXL3 linears of a file (a stem with a .trellis), routed experts or the others."""
    return {k[:-8] for k in t.h if k.endswith(".trellis") and bool(ROUTED.match(k)) == routed}


def official_tensors(bf16):
    """name -> (shard Tensors, key) for every tensor of the official checkpoint."""
    wm = json.load(open(os.path.join(bf16, "model.safetensors.index.json")))["weight_map"]
    files = {f: None for f in sorted(set(wm.values()))}

    def get(k):
        f = wm[k]
        if files[f] is None:
            files[f] = Tensors(os.path.join(bf16, f))
        return files[f]
    return wm, get


def write_nonexperts(sources, dst):
    """The non-expert EXL3 linears (attention projections, indexer.wq_b, shared experts, dense MLPs) of layers 0..77."""
    outs = []
    for t in sources:
        st = quantized_stems(t, routed=False)
        outs += [Out.copy(t, k) for k in t.data_order() if stem(k) in st]
    names = [o.name for o in outs]
    if len(set(names)) != len(names):
        sys.exit("non-expert linears: a tensor in two sources")
    write_tensors(dst, st_order(outs), {"format": "pt", "source": "EXL3 5-bit (mul1) non-expert linears of layers 0..77, exllamav3 1.6.1"})
    return len(outs)


def quant_config(experts, note=None):
    q = {"quant_method": "exl3", "exllamav3_version": "1.6.1",
         "routed_experts": {"bits": experts, "codebook": "mcg", "calibrated": "layers 3-77 yes, layer 78 (MTP) no",
                            "layout": "layers 3-77: cut into 8 tensor-parallel shares; layer 78 (MTP): whole"},
         "other_linears": {"bits": 5, "codebook": "mul1", "calibrated": "layers 0-77 yes, layer 78 (MTP) no"},
         "unquantized": f"embeddings, lm_head, norms, router, kv_b_proj, indexer wk / weights_proj / k_norm, eh_proj: as in zai-org/GLM-5.3-BF16 rev {REV}"}
    if note:
        q["routed_experts"]["note"] = note
    return q


def write_config(official_config, dst, qc):
    c = json.load(open(official_config))
    c.pop("quantization_config", None)          # a checkpoint's config.json as the base: its own description goes
    c["quantization_config"] = qc
    with open(dst + ".part", "w") as f:
        json.dump(c, f, indent=2)
        f.write("\n")
    os.replace(dst + ".part", dst)


def write_index(d):
    wm, total = {}, 0
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            t = Tensors(os.path.join(d, f))
            for k, v in t.h.items():
                if k in wm:
                    sys.exit(f"{k} in {wm[k]} and {f}")
                wm[k] = f
                total += v["data_offsets"][1] - v["data_offsets"][0]
            t.f.close()
    with open(os.path.join(d, "model.safetensors.index.json.part"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, f, indent=2)
    os.replace(os.path.join(d, "model.safetensors.index.json.part"), os.path.join(d, "model.safetensors.index.json"))
    return wm, total


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(PIECE)
            if not b:
                break
            h.update(b)
        drop_cache(f.fileno())
    return h.hexdigest()


def write_sums(d, jobs=4):
    names = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f != "SHA256SUMS"
                   and not f.startswith("README") and not f.endswith(".part"))
    with ProcessPoolExecutor(jobs) as ex:
        sums = list(ex.map(sha256_file, [os.path.join(d, f) for f in names]))
    with open(os.path.join(d, "SHA256SUMS"), "w") as f:
        f.writelines(f"{s}  {n}\n" for s, n in zip(sums, names))
    return len(names)


def place(src, dst, copy):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
    else:
        os.link(src, dst)


def step_nonexperts(qtensors, out):
    """nonexperts-5bit.safetensors from the non-expert conversion's per-layer files (layers 0..77)."""
    dst = os.path.join(out, "nonexperts-5bit.safetensors")
    if not os.path.exists(dst):
        srcs = [Tensors(os.path.join(qtensors, f"model.layers.{L}.safetensors")) for L in range(0, MTP)]
        n = write_nonexperts(srcs, dst)
        for t in srcs:
            t.close()
        log(f"nonexperts-5bit.safetensors: {n} tensors")


def step_mtp(layer78, linears, out):
    """The MTP layer's quantized files: its routed experts (the 4-bit conversion's model.layers.78.safetensors) and its
    other linears (convert_mtp_linears.py's file), each tensor as stored there."""
    q78, lin = Tensors(layer78), Tensors(linears)
    st78, stl = quantized_stems(q78, routed=True), quantized_stems(lin, routed=False)
    if len(st78) != 3 * EXPERTS or any(not x.startswith(f"model.layers.{MTP}.") for x in st78 | stl):
        sys.exit(f"{layer78} / {linears}: not the MTP layer's {3 * EXPERTS} routed expert linears and its other linears")
    for name, t, st, what in (
            ("mtp-layer-78-experts-4bit.safetensors", q78, st78,
             "EXL3 4-bit (mcg) routed experts of the MTP layer (78), exllamav3 1.6.1, uncalibrated"),
            ("mtp-layer-78-linears-5bit.safetensors", lin, stl,
             "EXL3 5-bit (mul1) linear layers of the MTP layer (78), exllamav3 1.6.1, uncalibrated")):
        dst = os.path.join(out, name)
        if not os.path.exists(dst):
            keys = [k for k in t.data_order() if stem(k) in st]
            write_tensors(dst, [Out.copy(t, k) for k in keys], {"format": "pt", "source": what})
            log(f"{name}: {len(keys)} tensors")
    q78.close(), lin.close()
    return st78 | stl


def step_unquantized(official, out):
    """The tensors kept unquantized: every official tensor whose linear has no quantized form in OUT, copied as stored."""
    wm, get = official_tensors(official)
    quant = {f"model.layers.{L}.mlp.experts.{e}.{p}" for L in range(FIRST, LAST + 1) for e in range(EXPERTS)
             for p in ("gate_proj", "up_proj", "down_proj")}
    for f in ("nonexperts-5bit.safetensors", "mtp-layer-78-experts-4bit.safetensors", "mtp-layer-78-linears-5bit.safetensors"):
        t = Tensors(os.path.join(out, f))
        quant |= quantized_stems(t, routed=True) | quantized_stems(t, routed=False)
        t.f.close()
    layer = lambda k: int(k.split(".")[2]) if k.startswith("model.layers.") else None
    keep = sorted(k for k in wm if stem(k) not in quant)
    main = [k for k in keep if layer(k) is None or layer(k) < MTP]
    mtp = [k for k in keep if layer(k) == MTP]
    other = [k for k in keep if k not in main and k not in mtp]
    if other:
        sys.exit(f"official tensors this layout has no place for: {other[:5]}")
    for name, keys, order, what in (
            ("unquantized-bf16.safetensors", main, list, "the tensors kept unquantized, copied as stored"),
            ("mtp-layer-78-unquantized-bf16.safetensors", mtp, st_order, "the MTP layer's (78) tensors kept unquantized, copied as stored")):
        dst = os.path.join(out, name)
        if not os.path.exists(dst):
            outs = [Out.copy(get(k), k) for k in keys]
            bad = [o.name for o in outs if o.dtype not in ("BF16", "F32")]
            if bad:
                sys.exit(f"{bad[:3]}: not BF16 (or F32) in the official files")
            n = write_tensors(dst, order(outs), {"format": "pt", "source": f"zai-org/GLM-5.3-BF16 rev {REV}: {what}"})
            log(f"{name}: {len(keys)} tensors, {n / 1e9:.2f} GB")
    return set(wm)


def step_small(official, out, qc):
    """The official small files as they are, and config.json with a quantization_config describing OUT."""
    for f in OFFICIAL_SMALL:
        shutil.copyfile(os.path.join(official, f), os.path.join(out, f))
    write_config(os.path.join(official, "config.json"), os.path.join(out, "config.json"), qc)


def cmd_checkpoint(a):
    os.makedirs(a.out, exist_ok=True)
    for L in range(FIRST, LAST + 1):
        f = f"experts-layer-{L:03d}.safetensors"
        place(os.path.join(a.experts, f), os.path.join(a.out, f), a.copy)
    log(f"experts: layers {FIRST}-{LAST} from {a.experts}")
    step_nonexperts(a.nonexperts, a.out)
    step_mtp(os.path.join(a.mtp_qtensors, f"model.layers.{MTP}.safetensors"), a.mtp_linears, a.out)
    names = step_unquantized(a.official, a.out)
    step_small(a.official, a.out, quant_config(4))
    finish(a.out, official=names)


def finish(d, official=None, same_as=None):
    """The index, then: every official tensor has its place (official), or the index is another's (same_as)."""
    wm, total = write_index(d)
    if official is not None:
        have = {re.sub(r"\.(trellis|suh|svh|mcg|mul1)$", ".weight", re.sub(r"\.rank\d+\.", ".", k)) for k in wm}
        missing = sorted(official - have)
        if missing:
            sys.exit(f"official tensors not in {d}: {missing[:5]} ({len(missing)})")
    if same_as is not None and wm != same_as:
        sys.exit(f"{d}: the index differs from the base checkpoint's")
    log(f"index: {len(wm)} tensors, {total / 1e9:.2f} GB")
    if cmd_check_dir(d):
        sys.exit("check failed")
    log(f"SHA256SUMS: {write_sums(d)} files")
    log(f"DONE {d}")


def mix_layer(base, bank3, sel, L, dst):
    """One layer of the four-node checkpoint: the selection's experts from the 3-bit bank, the rest from the base."""
    f = f"experts-layer-{L:03d}.safetensors"
    mem_guard()
    t4, t3 = Tensors(os.path.join(base, f)), Tensors(os.path.join(bank3, f))
    e_of = lambda k: int(ROUTED.match(k).group(2))
    if {e_of(k) for k in t4.h} != {e_of(k) for k in t3.h}:
        sys.exit(f"layer {L}: the two files hold different experts")
    pick = sel.get(L, set())
    outs = [Out.copy(t4, k) for k in t4.h if e_of(k) not in pick] + [Out.copy(t3, k) for k in t3.h if e_of(k) in pick]
    write_tensors(dst, st_order(outs), {"format": "pt", "mixed": f"layer {L}: {len(pick)} experts 3-bit"})
    t4.close(), t3.close()
    return f"layer {L}: {len(pick)} experts at 3 bits"


def cmd_mix(a):
    os.makedirs(a.out, exist_ok=True)
    sel = {int(k): set(v) for k, v in json.load(open(a.selection)).items()}
    for L in range(FIRST, LAST + 1):
        dst = os.path.join(a.out, f"experts-layer-{L:03d}.safetensors")
        if not os.path.exists(dst):
            log(mix_layer(a.base, a.bank3, sel, L, dst))
    for f in os.listdir(a.base):
        if f.startswith(("experts-layer-", "README", "SHA256SUMS", "config.json", "q3-selection")):
            continue
        place(os.path.join(a.base, f), os.path.join(a.out, f), a.copy)
    with open(os.path.join(a.out, "q3-selection.json"), "w") as f:
        json.dump({str(k): sorted(v) for k, v in sel.items()}, f)
    n3 = sum(len(v) for v in sel.values())
    write_config(os.path.join(a.base, "config.json"), os.path.join(a.out, "config.json"), quant_config(
        f"3/4 mix ({100 * n3 / (EXPERTS * (LAST - FIRST + 1)):g} %)",
        "routed experts of layers 3-77: those listed in q3-selection.json at 3 bits, the rest 4 bits; everything else as the eight-node checkpoint"))
    finish(a.out, same_as=json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"])


def cmd_check_dir(d):
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    by_file, bad = {}, 0
    for k, f in wm.items():
        by_file.setdefault(f, []).append(k)
    widths = {}
    for f, ks in sorted(by_file.items()):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print(f"missing file {f}")
            bad += 1
            continue
        t = Tensors(p)
        miss = [k for k in ks if k not in t.h]
        extra = [k for k in t.h if wm.get(k) != f]
        if miss or extra:
            print(f"{f}: {len(miss)} index keys not in the file, {len(extra)} file keys not indexed to it")
            bad += 1
        for k, v in t.h.items():
            if k.endswith(".trellis"):
                cb = "mcg" if k[:-8] + ".mcg" in t.h else "mul1" if k[:-8] + ".mul1" in t.h else "none"
                kind = "routed experts" if ROUTED.match(k) else "other linears"
                key = (kind, v["shape"][-1] // 16, cb)
                widths[key] = widths.get(key, 0) + 1
        t.f.close()
    for (kind, b, cb), n in sorted(widths.items()):
        print(f"{kind}: {n} trellis tensors at {b} bits, codebook {cb}")
    print(f"files {len(by_file)}, index keys {len(wm)}, problems {bad}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("qtensors"); b.add_argument("out")
    b.add_argument("--layers", default=f"{FIRST}-{LAST}"); b.add_argument("--jobs", type=int, default=2)
    c = sub.add_parser("checkpoint")
    for x in ("--official", "--experts", "--mtp-qtensors", "--nonexperts", "--mtp-linears", "--out"):
        c.add_argument(x, required=True)
    c.add_argument("--copy", action="store_true", help="copy the bank's expert files (default: hard links)")
    m = sub.add_parser("mix")
    for x in ("--base", "--bank3", "--selection", "--out"):
        m.add_argument(x, required=True)
    m.add_argument("--copy", action="store_true", help="copy the base's files (default: hard links)")
    k = sub.add_parser("check")
    k.add_argument("dir")
    a = p.parse_args()
    if a.cmd == "bank":
        cmd_bank(a)
    elif a.cmd == "checkpoint":
        cmd_checkpoint(a)
    elif a.cmd == "mix":
        cmd_mix(a)
    else:
        sys.exit(1 if cmd_check_dir(a.dir) else 0)


if __name__ == "__main__":
    main()
