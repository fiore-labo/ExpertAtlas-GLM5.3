#!/usr/bin/env python3
"""patch_experts.py CONVERT_MODEL_PY: exllamav3 1.6.1's convert_model.py patched to quantize only the routed experts
listed in env EXPERTS_LIST (JSON {"layer": [expert, ...]}; convert.sh lists every expert of layers 3..77) at the job's
--bits, taking every other linear from an earlier job's qtensors (env KEEP_QTENSORS: the 4-bit conversion, full4):

  - linears outside the list are not quantized; their tensors come from KEEP_QTENSORS, so the calibration state
    advances through the model with the 4-bit job's other linears and the new experts
  - the work dir's qtensors hold the listed experts' tensors only
  - expert calibration is exllamav3's own (activate_all_experts as in the full conversion), so a rarely routed expert
    still gets a Hessian from every calibration token
Fiore-lab's 3-bit bank was made this way. The patch checks the lines it changes and refuses another exllamav3 version.
"""
import sys
p = sys.argv[1]
s = open(p, encoding="utf8").read()

def sub(old, new, count):
    global s
    assert s.count(old) >= count, (old, s.count(old))
    s = s.replace(old, new, count)

helper = '''
# ---- patch_experts ----
import json as _json
from safetensors import safe_open as _safe_open
_KEEP_DIR = os.environ["KEEP_QTENSORS"]
_Q3 = {f"model.layers.{l}.mlp.experts.{e}." for l, es in _json.load(open(os.environ["EXPERTS_LIST"])).items() for e in es}
def _chosen(key):
    k = key + "."
    i = k.find(".mlp.experts.")
    if i < 0:
        return False
    j = k.find(".", i + len(".mlp.experts."))
    return k[:j + 1] in _Q3
def _keep(key):
    return not _chosen(key)
def _keep_tensors(module_key, prefixes):
    out = {}
    with _safe_open(os.path.join(_KEEP_DIR, module_key + ".safetensors"), "pt", device = "cpu") as f:
        for k in f.keys():
            if any(k.startswith(p) for p in prefixes):
                out[k] = f.get_tensor(k)
    return out
# ---- end ----

def check_system():'''
sub("\ndef check_system():", helper, 1)
sub("linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]",
    "linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None and not _keep(m.key)]", 1)
sub('''            # Unload module
            if not module.caps.get("retain_during_quant"):''',
    '''            # patch_experts: the kept linears' tensors (from KEEP_QTENSORS) in place of the unquantized ones
            _kl = [m.key + "." for m in module if isinstance(m, Linear) and m.qmap and _keep(m.key)]
            if _kl:
                for _k in [k for k in q_tensors if any(k.startswith(p) for p in _kl)]:
                    del q_tensors[_k]
                _old = _keep_tensors(module.key, _kl)
                assert all(any(k.startswith(p) for k in _old) for p in _kl), "kept linears missing in KEEP_QTENSORS"
                q_tensors.update(_old)
                print(f" -- Kept {len(_kl)} linears of {module.key} from {_KEEP_DIR}; quantized {len(linears)}", flush = True)

            # Unload module
            if not module.caps.get("retain_during_quant"):''', 1)
sub('save_tensor(q_tensors, f"qtensors/{module.key}.safetensors", args)',
    'save_tensor({k: v for k, v in q_tensors.items() if k.find(".mlp.experts.") >= 0 and _chosen(k.rsplit(".", 1)[0])}, f"qtensors/{module.key}.safetensors", args)', 1)
open(p, "w", encoding="utf8").write(s)
print("patched", p)
