#!/usr/bin/env bash
# Run and manage run.sh in a background tmux session.
#   bash task.sh start              # features, probe, head sweep
#   EVAL=1 bash task.sh start       # afterwards: select, val check, eval
#   bash task.sh status             # what is running, results so far, GPU
#   bash task.sh log                # follow job.log (Ctrl-C to quit)
#   bash task.sh attach             # enter the session (Ctrl-b d leaves)
#   bash task.sh stop               # interrupt
# Everything run.sh prints is appended to job.log; the session closes when
# run.sh ends and the last line of job.log gives its exit code.
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-qv-head}"
LOG=job.log

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
        [ -d "$d" ] || continue
        [ -f "${d}train-summary.json" ] || [ -f "${d}features.json" ] \
            || [ -f "${d}report.json" ] || basename "$d"
    done | grep -v '^aborted-' || true)
    if [ -n "$partial" ]; then
        echo "这些步骤没跑完，run.sh 会在它们那里停下："
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
        echo "== job.log 最后 5 行："
        tail -5 "$LOG"
    fi
    echo "== 各步骤："
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
    if (run / "features.json").is_file():
        record = json.loads((run / "features.json").read_text(encoding="utf-8"))
        print(f"{run.name}: 特征 train {record['n_train']} / val {record['n_val']}")
    elif (run / "train-summary.json").is_file():
        best = json.loads((run / "train-summary.json").read_text(encoding="utf-8"))
        best = best["best"]
        print(f"{run.name}: 完成，最佳第 {best['epoch']} 轮 "
              f"val 均值 {best['val_srcc_mean']:.3f}")
    elif (run / "report.json").is_file():
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        if "results" in report:
            best = max(report["results"], key=lambda r: r["val"]["srcc"])
            print(f"{run.name}: 探针，val 最高 {best['layers']} "
                  f"{best['dimension']} SRCC {best['val']['srcc']:.3f}")
        else:
            metrics = "  ".join(
                f"{k} SRCC {v['srcc']:.3f}" for k, v in report["metrics"].items()
            )
            print(f"{run.name}: {report['split']} 报告  {metrics}")
    else:
        history = run / "history.jsonl"
        lines = history.read_text(encoding="utf-8").splitlines() if history.is_file() else []
        print(f"{run.name}: 未完成，已完成 {len(lines)} 轮")
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
