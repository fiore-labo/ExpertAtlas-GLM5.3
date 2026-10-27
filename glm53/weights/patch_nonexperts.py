#!/usr/bin/env python3
"""patch_nonexperts.py CONVERT_MODEL_PY: exllamav3 1.6.1's convert_model.py patched to quantize only the non-expert
linears of a checkpoint whose routed experts are already converted (env KEEP_QTENSORS = that job's qtensors dir, the
4-bit conversion full4):

  - routed experts (model.layers.L.mlp.experts.E.*) are not quantized; their tensors are taken from KEEP_QTENSORS, so the
    calibration state advances through the final model (kept experts + the new non-expert linears)
  - calibration forwards run the routed experts only (activate_all_experts is for the experts' own Hessians)
  - the work dir's qtensors hold the non-expert tensors only (the experts are already in KEEP_QTENSORS)
Fiore-lab's nonexperts-5bit.safetensors was made this way. The patch checks the lines it changes and refuses another
exllamav3 version.
"""
import sys
p = sys.argv[1]
s = open(p, encoding="utf8").read()

def sub(old, new, count):
    global s
    assert s.count(old) >= count, (old, s.count(old))
    s = s.replace(old, new, count)

helper = '''
# ---- patch_nonexperts ----
import re as _re
from safetensors import safe_open as _safe_open
_KEEP_DIR = os.environ["KEEP_QTENSORS"]
_KEEP_RE = _re.compile(r"^model\\.layers\\.\\d+\\.mlp\\.experts\\.\\d+\\.")
_keep_cache = {}
def _keep(key):
    return _KEEP_RE.match(key + ".") is not None
def _keep_tensors(module_key):
    if module_key not in _keep_cache:
        _keep_cache.clear()
        out = {}
        with _safe_open(os.path.join(_KEEP_DIR, module_key + ".safetensors"), "pt", device = "cpu") as f:
            for k in f.keys():
                if _KEEP_RE.match(k):
                    out[k] = f.get_tensor(k)
        _keep_cache[module_key] = out
    return _keep_cache[module_key]
# ---- end ----

def check_system():'''
sub("\ndef check_system():", helper, 1)
sub("linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]",
    "linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None and not _keep(m.key)]", 1)
sub('''            # Unload module
            if not module.caps.get("retain_during_quant"):''',
    '''            # patch_nonexperts: the kept experts' tensors in place of the unquantized ones
            _kl = [m.key for m in module if isinstance(m, Linear) and _keep(m.key)]
            if _kl:
                for _k in [k for k in q_tensors if _KEEP_RE.match(k)]:
                    del q_tensors[_k]
                _old = _keep_tensors(module.key)
                assert all(any(k.startswith(l + ".") for k in _old) for l in _kl), "kept experts missing in KEEP_QTENSORS"
                q_tensors.update(_old)
                print(f" -- Kept {len(_kl)} expert linears of {module.key} from {_KEEP_DIR} ({len(_old)} tensors)", flush = True)

            # Unload module
            if not module.caps.get("retain_during_quant"):''', 1)
sub('save_tensor(q_tensors, f"qtensors/{module.key}.safetensors", args)',
    'save_tensor({k: v for k, v in q_tensors.items() if not _KEEP_RE.match(k)}, f"qtensors/{module.key}.safetensors", args)', 1)
n = s.count('"activate_all_experts": model.calibration_all_experts')
assert n >= 1, n
s = s.replace('"activate_all_experts": model.calibration_all_experts', '"activate_all_experts": False')
s = s.replace("if model.calibration_all_experts:", "if False:")
open(p, "w", encoding="utf8").write(s)
print("patched", p, "activate_all_experts sites:", n)
