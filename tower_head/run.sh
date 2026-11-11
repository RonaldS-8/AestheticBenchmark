#!/usr/bin/env bash
# The whole job on the cloud GPU, after prepare_data.py and job.py weights:
# cache train/val tower features, ridge probe, head sweep (LR x seed).
# With EVAL=1 it then selects on val, rescores the chosen head on val end to
# end (must match the cached-feature numbers) and scores eval once.
#   bash task.sh start           # runs this in tmux (preferred)
#   EVAL=1 bash task.sh start    # afterwards: select, val check, eval
# Finished steps are skipped on a rerun (features.json, report.json and
# train-summary.json mark them); a step that died half-way stops the job
# before it starts: rename it to runs/aborted-<id> or delete it, and rerun.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
# Python buffers output into pipes; without this tee and tail -F see nothing.
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/qwen35-4b-tower-head.toml}"
PREFIX="${PREFIX:-qwen35-4b-tower-head}"
LRS="${LRS:-1e-4 3e-4 1e-3}"
SEEDS="${SEEDS:-1 2 3}"
FEATURES="${PREFIX}-features"
mkdir -p runs

if [ -f "runs/${FEATURES}/features.json" ]; then
    echo "$(date '+%F %T') 跳过已完成的特征 ${FEATURES}"
else
    python job.py features --config "$CONFIG" --run-id "$FEATURES"
fi
if [ -f "runs/${PREFIX}-probe/report.json" ]; then
    echo "$(date '+%F %T') 跳过已完成的探针 ${PREFIX}-probe"
else
    python job.py probe --features "$FEATURES" --run-id "${PREFIX}-probe"
fi
# shellcheck disable=SC2086  # LRS and SEEDS are space-separated lists
python job.py sweep --config "$CONFIG" --features "$FEATURES" \
    --prefix "$PREFIX" --lrs $LRS --seeds $SEEDS

if [ "${EVAL:-0}" != "1" ]; then
    echo "$(date '+%F %T') 训练部分结束；看过 val 结果后用 EVAL=1 选模并评 eval"
    exit 0
fi
python job.py select --prefix "$PREFIX"
chosen=$(python -c "import json; print(json.load(open('runs/${PREFIX}-selection.json', encoding='utf-8'))['chosen_run_id'])")
python job.py evaluate --train-run "$chosen" --split val --run-id "${PREFIX}-val"
python job.py evaluate --train-run "$chosen" --split eval --run-id "${PREFIX}-eval"
echo "$(date '+%F %T') 全部完成"
