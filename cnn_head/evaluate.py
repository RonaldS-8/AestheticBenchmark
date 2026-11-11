"""Score a trained run's best checkpoint on val or eval (LoRA report format).

The eval split is for the final comparison only: score the one run chosen on
val, once. Output goes to ``runs/<run_id>/`` and is never overwritten.

Example::

    python evaluate.py --train-run convnext-t-ft-lr3e-5-s1 --split eval \\
        --run-id convnext-t-ft-eval
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aes_head.evaluation import evaluate_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, score, print the metrics."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--split", choices=("val", "eval"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)

    report = evaluate_run(
        data_root=args.data_root,
        runs_dir=args.runs_dir,
        train_run_id=args.train_run,
        split=args.split,
        run_id=args.run_id,
        device=args.device,
        precision=args.precision,
        now=datetime.now().astimezone(),
    )
    for dim, m in report.metrics.items():
        print(f"{dim}: SRCC {m.srcc:.3f}  PLCC {m.plcc:.3f}  MAE {m.mae_1_5:.3f}")
    print(
        f"每张 p50 {report.latency_p50_ms:.0f} ms，p95 {report.latency_p95_ms:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
