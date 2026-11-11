"""Fine-tune a backbone with two score heads on EVA (one run).

The recipe comes from a TOML file; the seed and backbone learning rate are
per run. ``--dry-run`` checks data, weights and runtime and runs one forward
pass, writing nothing. The other overrides are for local smoke tests.

Example::

    python train.py --config configs/convnext-t-ft.toml \\
        --run-id convnext-t-ft-lr3e-5-s1 --seed 1 --backbone-lr 3e-5 --dry-run
    python train.py --config configs/convnext-t-ft.toml \\
        --run-id convnext-t-ft-lr3e-5-s1 --seed 1 --backbone-lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aes_head.config import apply_overrides, load_config  # noqa: E402
from aes_head.finetune import dry_run, finetune  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, then dry-run or train."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone-lr", type=float, required=True)
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--models-dir", type=Path, default=HERE / "models")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    args = parser.parse_args(argv)

    runtime = {
        "device": args.device,
        "precision": args.precision,
        "num_workers": args.num_workers,
    }
    train = {
        "batch_size": args.batch_size,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
    }
    config = apply_overrides(
        load_config(args.config),
        {
            "runtime": {k: v for k, v in runtime.items() if v is not None},
            "train": {k: v for k, v in train.items() if v is not None},
        },
    )
    common = {
        "data_root": args.data_root,
        "models_dir": args.models_dir,
        "runs_dir": args.runs_dir,
        "run_id": args.run_id,
        "limit": args.limit,
    }
    if args.dry_run:
        print(json.dumps(dry_run(config, **common), ensure_ascii=False, indent=2))
        return 0
    summary = finetune(
        config,
        seed=args.seed,
        backbone_lr=args.backbone_lr,
        now=datetime.now().astimezone(),
        **common,
    )
    best = summary.best
    print(
        f"{summary.run_id}: 最佳 {best.stage} 第 {best.epoch} 轮，val SRCC 均值 "
        f"{best.val_srcc_mean:.3f}，用时 {summary.seconds / 60:.1f} 分钟"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
