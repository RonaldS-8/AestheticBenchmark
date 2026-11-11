#!/usr/bin/env bash
# Create .venv for the Qwen vision-tower head job and check the GPU.
# Run from anywhere inside the unpacked bundle:
#   bash setup.sh
#   PYTHON=/path/to/python3.11 bash setup.sh      # pick the interpreter
# On Linux it must end with CUDA available and bf16 supported, or it fails.
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
import torchvision
import transformers

print({"python": sys.version.split()[0], "torch": torch.__version__,
       "cuda": torch.version.cuda, "torchvision": torchvision.__version__,
       "transformers": transformers.__version__})
if sys.platform != "linux":
    print("非 Linux：CPU 版 torch，只能做冒烟测试")
    raise SystemExit(0)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用：检查驱动是否支持 CUDA 12")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("这块 GPU 不支持 bf16，配置里改用 fp32")
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({"gpus": names})
PY
