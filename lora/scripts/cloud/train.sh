#!/usr/bin/env bash
# Full cloud job: preflight, LoRA training, LoRA eval on val and eval.
#   bash scripts/cloud/train.sh configs/eva-qwen35-4b-lora-r16.toml
# The baseline "same prompt, no LoRA" is not run here: prompt tuning already
# scored it (ADR-0012), and scripts/report_lora_gain.py compares the two.
# Safe to re-run after any interruption:
# - evals resume per image; finished ones are skipped;
# - training refuses an existing output dir unless RESUME=1, which continues
#   from the newest complete checkpoint;
# - once train_result.json exists, training is done and is skipped.
# Set HF_ENDPOINT for a Hugging Face mirror if needed.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CONFIG="${1:?用法: train.sh <config.toml>}"
read_config() {  # read_config <python expression over the parsed TOML c>
    python -c "import sys,tomllib,pathlib;c=tomllib.load(open(sys.argv[1],'rb'));print($1)" "$CONFIG"
}
RUN_NAME=$(read_config "c['run_name']")
NUM_GPUS=$(read_config "c['runtime']['num_gpus']")
OUTPUT_DIR=$(read_config "c['runtime']['output_dir']")
# Samples come from the config, so eval uses the same prompt as training.
SAMPLES_DIR=$(read_config "pathlib.PurePosixPath(c['data']['train_samples']).parent")
LOG_DIR="$OUTPUT_DIR.logs"
mkdir -p "$LOG_DIR"

TRAIN=(scripts/train_lora_sft.py --config "$CONFIG" --eval-samples "$SAMPLES_DIR/eval.jsonl")
evaluate() {  # evaluate <split> <run-id>
    python scripts/eval_eva_sft.py --config "$CONFIG" \
        --samples "$SAMPLES_DIR/$1.jsonl" --split "$1" --run-id "$2" \
        --adapter "$OUTPUT_DIR/adapter" 2>&1 | tee -a "$LOG_DIR/eval-$2.log"
}

TRAINED=0
if [ -f "$OUTPUT_DIR/train_result.json" ]; then TRAINED=1; fi

echo "== 1/3 preflight"
if [ "$TRAINED" = "1" ]; then
    echo "训练已完成（$OUTPUT_DIR/train_result.json），跳过"
else
    python "${TRAIN[@]}" --preflight 2>&1 | tee -a "$LOG_DIR/preflight.log"
fi

echo "== 2/3 train"
EXTRA=()
if [ "${RESUME:-0}" = "1" ]; then EXTRA+=(--resume); fi
if [ "$TRAINED" = "1" ]; then
    echo "训练已完成，跳过"
elif [ "$NUM_GPUS" = "1" ]; then
    python "${TRAIN[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
else
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "${TRAIN[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
fi

echo "== 3/3 LoRA eval (val, then the frozen eval set for the gain report)"
evaluate val "${RUN_NAME}-val"
evaluate eval "${RUN_NAME}-eval"
echo "完成：把 $OUTPUT_DIR 和 data/runs/${RUN_NAME}-val、-eval 下载回本机，"
echo "再按 docs/cloud-training.md 用 scripts/report_lora_gain.py 算提升"
