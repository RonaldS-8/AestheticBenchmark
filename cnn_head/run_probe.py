"""Frozen-feature probe: which backbone predicts EVA scores best at native size.

Extracts multi-level pooled features of train and val images (never eval),
fits a ridge regression per score with the strength chosen by cross-validation
on train, and reports val SRCC/PLCC/MAE. Features are cached per backbone in
``runs/<run_id>/`` so an interrupted run resumes.

Example::

    python run_probe.py --run-id probe-native-v1
    python run_probe.py --run-id probe-smoke --limit 40
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aes_head.backbones import BACKBONES  # noqa: E402
from aes_head.probe import DEFAULT_ALPHAS  # noqa: E402
from aes_head.run import format_report, run_probe  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the probe and print the table."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--models-dir", type=Path, default=HERE / "models")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    report = run_probe(
        data_root=args.data_root,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        backbones=args.backbones,
        device=args.device,
        limit=args.limit,
        folds=args.folds,
        alphas=DEFAULT_ALPHAS,
        bootstrap=args.bootstrap,
        seed=args.seed,
        now=datetime.now().astimezone(),
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
