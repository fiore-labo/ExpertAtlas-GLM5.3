#!/usr/bin/env python3
"""convert_mtp_linears.py MODEL_DIR OUT_FILE [--bits 5] [--device 0]: the MTP layer (model.layers.78) of the GLM-5.3 BF16
checkpoint with its non-expert linears at BITS (mul1 codebook) and eh_proj kept BF16; the routed experts are skipped
(the 4-bit conversion's mcg ones are used as they are). exllamav3 1.6.1's util/convert_mtp.py, but with one codebook
and width per kind of linear instead of one for all, uncalibrated like it. Fiore-lab's mtp-layer-78-linears-5bit was
made this way; assemble.py keeps the file's quantized linears and takes the rest from the official files."""
import argparse, os, sys, time
import torch
from exllamav3 import Model, Config
from exllamav3.modules import Linear
from exllamav3.modules.quant import LinearEXL3, LinearFP16
from exllamav3.loader.safetensors_alt import save_file

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out_file")
ap.add_argument("--bits", type=int, default=5); ap.add_argument("--device", type=int, default=0)
a = ap.parse_args()

config = Config.from_directory(a.model_dir)
mtp = Model.from_config(config, component="mtp")
mtp.load(progressbar=True)
out = {}
for module in mtp.modules:
    module.load(torch.device("cpu" if module.caps.get("prefer_cpu") else a.device))
    module.config.stc.close()
    linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]
    keep = set()
    for lin in linears:
        lin.inner.swap_cpu()
        if ".mlp.experts." in lin.key:                       # the full conversion's 4-bit mcg experts stay
            keep.add(lin.key); continue
        if lin.key.endswith("eh_proj"):                      # small; BF16
            print(f" -- kept BF16: {lin.key}", flush=True); continue
        t0 = time.time()
        err = lin.convert_exl3(lin.init_H_data(False), quant_args={"seed": 0, "K": a.bits, "mul1": True,
                               "devices": [a.device], "device_ratios": None, "apply_out_scales": "always"},
                               progress_str=f" -- <step>: {lin.key}", verbose=False, save_reg=None)
        assert isinstance(lin.inner, LinearEXL3)
        lin.inner.swap_cpu()
        print(f" -- quantized {lin.key}: {a.bits} bits mul1, proxy_err {err:.6f} [{time.time() - t0:.1f} s]", flush=True)
    for m in module:
        if m.key not in keep:
            out.update({k: v for k, v in m.get_tensors().items() if ".mlp.experts." not in k})
    module.unload()
    config.stc.close()
save_file(out, os.path.abspath(a.out_file))
print(f" -- wrote {len(out)} tensors to {a.out_file}")
