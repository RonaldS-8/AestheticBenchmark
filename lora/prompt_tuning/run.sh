#!/usr/bin/env bash
# Full prompt-tuning job: B0, B1, B2 over the tuning set, then pick the best.
#   SOURCE=hf HF_ENDPOINT=https://hf-mirror.com bash run.sh   # HF 镜像
#   SOURCE=modelscope bash run.sh                              # ModelScope
#   SOURCE=hf DEVICE=cuda:1 TAG=qwen35-4b bash run.sh
# Safe to re-run after any interruption: every run resumes per image, and a
# finished selection (runs/selection.json) is left as it is.
# SOURCE (hf or modelscope) is required; weights are checked file by file.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

SOURCE="${SOURCE:?设置 SOURCE=hf（可加 HF_ENDPOINT 镜像）或 SOURCE=modelscope}"
DEVICE="${DEVICE:-cuda}"
TAG="${TAG:-qwen35-4b}"
mkdir -p logs runs

echo "== 权重（只在第一次真正下载）"
python download_model.py --source "$SOURCE" 2>&1 | tee -a logs/download.log

run() {  # run <method> <prompt-id> <short-name>
    local run_id="$3-$TAG-tune"
    echo "== $run_id ($2)"
    python run_baseline.py --method "$1" --prompt-id "$2" \
        --run-id "$run_id" --device "$DEVICE" 2>&1 | tee -a "logs/$run_id.log"
}

run b0_zero_shot zero-shot-score@v2 b0
run b1_rubric rubric-score@v2 b1
run b2_few_shot few-shot-score@v2 b2

if [ -f runs/selection.json ]; then
    echo "runs/selection.json 已存在，跳过选择"
else
    python select_best_prompt.py --output runs/selection.json \
        "runs/b0-$TAG-tune" "runs/b1-$TAG-tune" "runs/b2-$TAG-tune" \
        2>&1 | tee -a logs/selection.log
fi
echo "完成：把 runs/ 整个下载回来"
