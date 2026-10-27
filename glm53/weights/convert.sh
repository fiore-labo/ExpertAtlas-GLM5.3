#!/bin/bash
# convert.sh STEP: zai-org/GLM-5.3-BF16 -> the EXL3 parts of this recipe's weights, with exllamav3 1.6.1 (README.md).
#   setup          a Python venv ($VENV) with PyTorch 2.8 (CUDA 12.8) and exllamav3 1.6.1's release wheel: x86_64
#                  Linux with NVIDIA GPUs (a DGX Spark is aarch64: README.md, "On DGX Spark")
#   full4          every linear at 4 bits (codebook mcg), calibrated with exllamav3's default calibration data: its
#                  routed experts (layers 3..77, and the MTP layer's) are the 4-bit bank; its other linears only
#                  serve the steps below. Work directory $WORK/full4
#   experts BITS   the routed experts of layers 3..77 at BITS (2, 3 or 5; mcg), calibrated through the model whose
#                  other linears are full4's (patch_experts.py). Needs full4. Work directory $WORK/experts$BITS
#   nonexperts     the other linears of layers 0..77 at 5 bits (codebook mul1), calibrated through the model with
#                  full4's routed experts (patch_nonexperts.py). Needs full4. Work directory $WORK/nonexperts
#   mtp            the MTP layer's other linears at 5 bits (mul1), without calibration data, as exllamav3 converts an
#                  MTP layer (convert_mtp_linears.py, one GPU). Writes $WORK/mtp-linears-5bit.safetensors
# Settings (environment): BF16 = the official checkpoint, checked with check_official.py (required); WORK (default
# ./work); GPUS = CUDA devices, e.g. 0,1,2,3 (default: every GPU nvidia-smi lists); VENV (default ./venv); PYTHON =
# the Python 3.12 that setup makes the venv with (default python3; the ExLlamaV3 wheel is for 3.12).
# The conversion steps patch the venv's copy of ExLlamaV3's converter as they start (restored from the original each
# time): run one step at a time per venv.
# A step runs in the foreground (use tmux or nohup for hours-long steps) and logs to $WORK/<step>.log. Started again
# after an interruption, it resumes from its last checkpoint (exllamav3's -r). exllamav3 also writes a complete model
# of its own to $WORK/<step>-out; this recipe does not use it (assemble.py reads <step>/qtensors), so delete it when
# the step is done if the disk is tight.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STEP=${1:-}
WORK=${WORK:-$PWD/work}
VENV=${VENV:-$PWD/venv}
PY=$VENV/bin/python
WHEEL="https://github.com/turboderp-org/exllamav3/releases/download/v1.6.1/exllamav3-1.6.1%2Bcu128.torch2.8.0-cp312-cp312-linux_x86_64.whl"

usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

need_bf16() {
  [ -n "${BF16:-}" ] && [ -f "$BF16/config.json" ] && [ -f "$BF16/model.safetensors.index.json" ] \
    || { echo "BF16 must name the official checkpoint's directory (zai-org/GLM-5.3-BF16, checked with check_official.py)"; exit 2; }
  [ -x "$PY" ] || { echo "no venv at $VENV: run 'convert.sh setup' first"; exit 2; }
  mkdir -p "$WORK"
}

devices() {
  if [ -n "${GPUS:-}" ]; then echo "$GPUS"; else seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )); fi
}

convert_model() {           # convert_model NAME ARGS...: exllamav3's converter, resumed when NAME's work dir has started
  local name=$1; shift
  local w=$WORK/$name resume=()
  mkdir -p "$w" "$w-out"
  [ -f "$w/args.json" ] && resume=(-r)
  echo "$(date '+%F %T') $name: $* on GPUs $(devices) ${resume[*]}" | tee -a "$WORK/$name.log"
  "$PY" -m exllamav3.conversion.convert_model -i "$BF16" -w "$w" -o "$w-out" "$@" -d "$(devices)" -cpi 1800 -v \
    ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "$WORK/$name.log"
  echo "$(date '+%F %T') $name: done, qtensors in $w/qtensors" | tee -a "$WORK/$name.log"
}

patched() {                 # patched PATCH|none: the venv's convert_model.py as released, with PATCH applied
  local patch=$1 cm
  cm=$("$PY" -c 'import exllamav3, os; print(os.path.join(os.path.dirname(exllamav3.__file__), "conversion", "convert_model.py"))')
  [ -f "$cm.orig" ] || cp "$cm" "$cm.orig"
  cp "$cm.orig" "$cm"
  [ "$patch" = none ] || "$PY" "$HERE/$patch" "$cm"
}

case "$STEP" in
  setup)
    "${PYTHON:-python3}" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is needed (the ExLlamaV3 1.6.1 wheel is cp312): set PYTHON")'
    "${PYTHON:-python3}" -m venv "$VENV"
    "$PY" -m pip install -q -U pip
    "$PY" -m pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    "$PY" -m pip install -q "$WHEEL" numpy xxhash
    "$PY" -c 'import torch, exllamav3; print("exllamav3", exllamav3.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")'
    ;;
  full4)
    need_bf16
    patched none
    convert_model full4 -b 4 -hb 16 -cb mcg
    ;;
  experts)
    BITS=${2:-}
    [[ $BITS =~ ^[235]$ ]] || { echo "experts BITS: 2, 3 or 5 (4 bits is full4)"; exit 2; }
    need_bf16
    [ -d "$WORK/full4/qtensors" ] || { echo "run full4 first"; exit 2; }
    [ -f "$WORK/experts-all.json" ] || "$PY" -c "import json; json.dump({str(l): list(range(256)) for l in range(3, 78)}, open('$WORK/experts-all.json', 'w'))"
    patched patch_experts.py
    KEEP_QTENSORS=$WORK/full4/qtensors EXPERTS_LIST=$WORK/experts-all.json convert_model "experts$BITS" -b "$BITS" -hb 16 -mb 4 -cb mcg
    ;;
  nonexperts)
    need_bf16
    [ -d "$WORK/full4/qtensors" ] || { echo "run full4 first"; exit 2; }
    patched patch_nonexperts.py
    KEEP_QTENSORS=$WORK/full4/qtensors convert_model nonexperts -b 5 -hb 16 -mb 16 -cb mul1
    ;;
  mtp)
    need_bf16
    dev=$(devices); dev=${dev%%,*}
    "$PY" "$HERE/convert_mtp_linears.py" "$BF16" "$WORK/mtp-linears-5bit.safetensors" --bits 5 --device "$dev" 2>&1 | tee -a "$WORK/mtp.log"
    ;;
  *) usage ;;
esac
