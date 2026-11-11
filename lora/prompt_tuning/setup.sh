#!/usr/bin/env bash
# Create .venv for the prompt-tuning job and check the GPU stack.
# Run from anywhere inside the unpacked bundle:
#   bash setup.sh
#   PYTHON=/path/to/python3.11 bash setup.sh      # pick the interpreter
# On Linux it must end with CUDA 12 and both linear-attention kernels
# importable, or it fails.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
check_python='import sys
if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit(f"需要 Python 3.11–3.13，当前 {sys.version.split()[0]}")'
"$PYTHON" -c "$check_python"
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi
python -c "$check_python"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import sys

import torch
import transformers

print({"python": sys.version.split()[0], "torch": torch.__version__,
       "cuda": torch.version.cuda, "transformers": transformers.__version__})
if sys.platform != "linux":
    print("非 Linux：CPU 版 torch，只能做冒烟测试")
    raise SystemExit(0)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用：检查驱动是否支持 CUDA 12")
import importlib

# Same kernels the main repository's trainer requires.
kernels = (
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    ("causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d_update"),
)
missing = []
for module_name, function in kernels:
    try:
        getattr(importlib.import_module(module_name), function)
    except Exception as exc:
        missing.append(f"{module_name}.{function}: {exc}")
if missing:
    raise SystemExit(f"加速内核不可用: {missing}")
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({"gpus": names, "bf16": torch.cuda.is_bf16_supported()})
PY
