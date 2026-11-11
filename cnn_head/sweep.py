"""Train a learning-rate x seed sweep in one process (see aes_head/sweep.py).

Run ids are ``<prefix>-lr<lr>-s<seed>``; finished runs are skipped. Each run
also writes ``logs/<run_id>.log``.

Example::

    python sweep.py --config configs/convnext-t-ft.toml --prefix convnext-t-ft \\
        --lrs 1e-5 3e-5 1e-4 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aes_head.config import load_config  # noqa: E402
from aes_head.sweep import run_sweep  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and train every unfinished run of the sweep."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--lrs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--models-dir", type=Path, default=HERE / "models")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--logs-dir", type=Path, default=HERE / "logs")
    args = parser.parse_args(argv)

    summaries = run_sweep(
        load_config(args.config),
        prefix=args.prefix,
        lrs=args.lrs,
        seeds=args.seeds,
        data_root=args.data_root,
        models_dir=args.models_dir,
        runs_dir=args.runs_dir,
        logs_dir=args.logs_dir,
        clock=lambda: datetime.now().astimezone(),
    )
    print(f"sweep 完成：共 {len(summaries)} 组", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
