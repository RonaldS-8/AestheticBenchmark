#!/usr/bin/env bash
# Create the shared .venv with the pinned training + dev stack and check it.
# Same requirements on every machine; nothing is compiled. Run from anywhere:
#   bash scripts/cloud/setup.sh                      # cloud GPU (Linux)
#   PYTHON=/path/to/python3.11 bash scripts/cloud/setup.sh   # pick Python
# On Linux the run must end with CUDA 12 and every kernel importable, or it
# fails. Elsewhere (the Windows dev machine) torch is the CPU build: tests,
# dry-run and preflight run, training does not.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
check_python='import sys
ok = (3, 11) <= sys.version_info[:2] <= (3, 13)
if not ok:
    raise SystemExit(
        f"需要 Python 3.11–3.13（预编译 CUDA 内核只覆盖到 3.13），"
        f"当前 {sys.version.split()[0]}"
    )'
"$PYTHON" -c "$check_python"
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi
python -c "$check_python"  # an existing .venv may be on another Python
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt -r requirements-dev.txt
python - <<'PY'
import sys

import torch

from lora_sft.train.qwen_lora import kernel_status

status = kernel_status()
print({"python": sys.version.split()[0], "torch": torch.__version__,
       "cuda": torch.version.cuda, "linear_attention_kernels": status})
if sys.platform != "linux":
    print("非 Linux：CPU 版 torch，可跑测试 / dry-run / preflight，不能训练")
    raise SystemExit(0)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用：检查驱动是否支持 CUDA 12")
if not (torch.version.cuda or "").startswith("12."):
    raise SystemExit(f"需要 CUDA 12 版 torch，当前 {torch.version.cuda}")
missing = {name: error for name, error in status.items() if error is not None}
if missing:
    raise SystemExit(f"加速内核不可用: {missing}")
import flash_attn

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({"flash_attn": flash_attn.__version__, "gpus": names,
       "bf16": torch.cuda.is_bf16_supported()})
PY
