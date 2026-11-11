#!/usr/bin/env bash
# Fine-tuning sweep on the cloud GPU: backbone LR x seed, choose on val,
# then score the chosen run once on eval.
#   nohup bash run.sh > sweep.log 2>&1 &     # survives SSH disconnects
#   tail -F sweep.log                         # watch progress
#   LRS="3e-5" SEEDS="1" bash run.sh          # a single run
#   EVAL=1 bash run.sh                        # finally: select on val, score eval
# Selection and the eval split run only with EVAL=1, so trial runs never
# touch eval; finished runs are skipped, so EVAL=1 after the sweep only
# selects and scores. Prefer task.sh, which runs this in tmux.
# All runs train in one Python process (sweep.py), so the per-image-size
# cuDNN warm-up is paid once, not once per run. Each run also writes
# logs/<run_id>.log and runs/<run_id>/history.jsonl (one line per finished
# epoch). Finished runs (train-summary.json present) are skipped on a rerun;
# a run that died half-way stops the sweep before it starts: check it, then
# rename it to runs/aborted-<run_id> or delete it, and rerun.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
# Python buffers output into pipes; without this tee and tail -F see nothing.
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/convnext-t-ft.toml}"
PREFIX="${PREFIX:-convnext-t-ft}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p logs runs

# shellcheck disable=SC2086  # LRS and SEEDS are space-separated lists
python sweep.py --config "$CONFIG" --prefix "$PREFIX" --lrs $LRS --seeds $SEEDS

if [ "${EVAL:-0}" != "1" ]; then
    echo "$(date '+%F %T') 训练部分结束；全部运行完成后用 EVAL=1 选模并评 eval"
    exit 0
fi
python select_run.py --prefix "$PREFIX" 2>&1 | tee "logs/${PREFIX}-selection.log"
chosen=$(python -c "import json; print(json.load(open('runs/${PREFIX}-selection.json', encoding='utf-8'))['chosen_run_id'])")
python evaluate.py --train-run "$chosen" --split eval \
    --run-id "${PREFIX}-eval" 2>&1 | tee "logs/${PREFIX}-eval.log"
echo "$(date '+%F %T') 全部完成"
