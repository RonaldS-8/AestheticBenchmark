"""Fix the EVA prompt-tuning set: few-shot examples and the images to score.

All of EVA is scored except ``k`` training images picked by aesthetic
quantiles, which B2 shows as examples (ADR-0012). Writes
``data/manifests/eva/prompt-tuning/<version>.json``; an existing file is
never overwritten.

Example::

    python scripts/prepare_prompt_tuning.py \\
        --sft-manifest manifests/eva/sft/v1/manifest.json --k 3 --version v1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import SHANGHAI_TIMEZONE  # noqa: E402
from lora_sft.crawl.http import sha256_file  # noqa: E402
from lora_sft.eval.eva_baseline import build_tuning_set, write_tuning_set  # noqa: E402
from lora_sft.train.eva_data import load_eva_samples  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Build the tuning set and write it (or only print it with ``--dry-run``)."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft-manifest", required=True, help="相对 data 根目录")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.data_root / args.sft_manifest
    splits = {
        name: load_eva_samples(manifest_path.parent / f"{name}.jsonl", name)
        for name in ("train", "val", "eval")
    }
    tuning_set = build_tuning_set(
        splits["train"],
        splits["val"],
        splits["eval"],
        k=args.k,
        version=args.version,
        sft_manifest=Path(args.sft_manifest).as_posix(),
        sft_manifest_sha256=sha256_file(manifest_path),
        now=datetime.now(SHANGHAI_TIMEZONE),
    )
    output = args.data_root / "manifests" / "eva" / "prompt-tuning"
    output = output / f"{args.version}.json"
    print(tuning_set.model_dump_json(indent=2))
    if args.dry_run:
        print(f"dry run：不写入 {output}")
    else:
        write_tuning_set(output, tuning_set)
        print(f"已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
