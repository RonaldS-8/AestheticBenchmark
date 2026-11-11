#!/usr/bin/env bash
# Run and manage the fine-tuning sweep in a background tmux session.
#   bash task.sh start                      # whole sweep (run.sh defaults)
#   LRS="3e-5" SEEDS="1" bash task.sh start # pass run.sh settings through
#   EVAL=1 bash task.sh start               # after the sweep: select, score eval
#   bash task.sh status                     # what is running, progress, GPU
#   bash task.sh log                        # follow sweep.log (Ctrl-C to quit)
#   bash task.sh attach                     # enter the session (Ctrl-b d leaves)
#   bash task.sh stop                       # interrupt the sweep
# Everything run.sh prints is appended to sweep.log; the session closes when
# run.sh ends and the last line of sweep.log gives its exit code.
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-aes-head}"
LOG=sweep.log

running() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

start() {
    if running; then
        echo "tmux 会话 ${SESSION} 已在运行；先 bash task.sh status 看看"
        exit 1
    fi
    local partial
    partial=$(for d in runs/*/; do
        [ -d "$d" ] && [ ! -f "${d}train-summary.json" ] && basename "$d"
    done | grep -v -e '-eval$' -e '^aborted-' || true)
    if [ -n "$partial" ]; then
        echo "这些运行没跑完，run.sh 会在它们那里停下："
        echo "$partial"
        echo "确认后改名留存（mv runs/<id> runs/aborted-<id>）或删除，再启动"
        exit 1
    fi
    local env_args=()
    for name in LRS SEEDS CONFIG PREFIX EVAL; do
        if [ -n "${!name:-}" ]; then
            env_args+=(-e "${name}=${!name}")
        fi
    done
    echo "$(date '+%F %T') task.sh start LRS='${LRS:-默认}'" \
        "SEEDS='${SEEDS:-默认}' EVAL='${EVAL:-0}'" >> "$LOG"
    local command="bash run.sh 2>&1 | tee -a $LOG; code=\${PIPESTATUS[0]}; \
echo \"\$(date '+%F %T') run.sh 退出码 \$code\" | tee -a $LOG"
    tmux new-session -d -s "$SESSION" -c "$PWD" "${env_args[@]}" bash -c "$command"
    echo "已在 tmux 会话 ${SESSION} 里启动；看进度：bash task.sh log 或 status"
}

status() {
    if running; then
        echo "== tmux 会话 ${SESSION}：运行中"
    else
        echo "== tmux 会话 ${SESSION}：没有在跑"
    fi
    if [ -f "$LOG" ]; then
        echo "== 最近一组："
        grep -E '^\[[0-9]+/[0-9]+\]' "$LOG" | tail -1 || true
        echo "== 最新进度："
        grep -E '^\[(head|full) [0-9]+/[0-9]+\]' "$LOG" | tail -3 || true
        echo "== sweep.log 最后一行："
        tail -1 "$LOG"
    fi
    echo "== 各运行（每轮的 val SRCC 均值）："
    if [ ! -x .venv/bin/python ]; then
        echo "没有 .venv，先运行 bash setup.sh"
        exit 1
    fi
    .venv/bin/python - <<'PY'
import json
from pathlib import Path

runs = sorted(p for p in Path("runs").glob("*") if p.is_dir())
if not runs:
    print("（还没有）")
for run in runs:
    summary = run / "train-summary.json"
    if summary.is_file():
        best = json.loads(summary.read_text(encoding="utf-8"))["best"]
        print(f"{run.name}: 完成，最佳 {best['stage']} {best['epoch']} "
              f"val 均值 {best['val_srcc_mean']:.3f}")
        continue
    if (run / "report.json").is_file():
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        metrics = "  ".join(
            f"{k} SRCC {v['srcc']:.3f}" for k, v in report["metrics"].items()
        )
        print(f"{run.name}: {report['split']} 报告  {metrics}")
        continue
    history = run / "history.jsonl"
    lines = history.read_text(encoding="utf-8").splitlines() if history.is_file() else []
    trail = " ".join(
        f"{e['stage']}{e['epoch']}:{e['val_srcc_mean']:.3f}"
        for e in map(json.loads, lines)
    )
    print(f"{run.name}: 未完成，已完成 {len(lines)} 轮 {trail}")
PY
    echo "== GPU："
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
        --format=csv,noheader
}

case "${1:-}" in
    start) start ;;
    status) status ;;
    log) tail -n 30 -F "$LOG" ;;
    attach) tmux attach -t "$SESSION" ;;
    stop)
        if ! running; then
            echo "没有在跑"
            exit 0
        fi
        tmux send-keys -t "$SESSION" C-c
        for _ in $(seq 30); do
            running || break
            sleep 1
        done
        if running; then
            tmux kill-session -t "$SESSION"
        fi
        echo "$(date '+%F %T') task.sh stop" >> "$LOG"
        echo "已停止；没跑完的运行目录还在 runs/ 下，重启前先处理"
        ;;
    *)
        echo "用法: bash task.sh {start|status|log|attach|stop}"
        exit 1
        ;;
esac
