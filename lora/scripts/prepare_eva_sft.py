"""Prepare immutable EVA JSONL messages for Qwen3.5-4B LoRA SFT."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.data.eva_sft import SplitRatios, prepare_eva_sft  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the EVA SFT preparation command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("prompts/eva-sft-v1/eva-dual-score.prompt.md"),
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate inputs, preview the split and optionally write it."""

    args = parse_args()
    ratios = SplitRatios(
        train=args.train_ratio,
        val=args.val_ratio,
        eval=args.eval_ratio,
    )
    plan = prepare_eva_sft(
        args.data_root,
        args.prompt,
        version=args.version,
        seed=args.seed,
        ratios=ratios,
        created_at=datetime.now(SHANGHAI_TIMEZONE),
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "dataset": "eva",
                "version": plan.manifest.version,
                "model": plan.manifest.qwen_input.model_id,
                "output_contract": plan.manifest.output_contract,
                "record_count": plan.manifest.record_count,
                "train_count": plan.manifest.train_count,
                "val_count": plan.manifest.val_count,
                "eval_count": plan.manifest.eval_count,
                "eval_fingerprint": plan.manifest.eval_fingerprint,
                "output_dir": plan.output_dir.as_posix(),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
