"""Crawl the public EVA score table and optional image archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lora_sft.crawl.eva_runner import crawl_eva  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the EVA crawler command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="数据根目录，默认 ./data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只抓取并校验评分表，不写文件",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="下载约 700 MB 分卷并严格校验后解压图片",
    )
    parser.add_argument(
        "--reuse-existing-images",
        action="store_true",
        help="恢复运行时复用已完整下载的 7 个分卷；仍重新抓取评分表",
    )
    return parser.parse_args()


def main() -> None:
    """Run the crawler and print a machine-readable summary."""

    args = parse_args()
    result = crawl_eva(
        args.data_root,
        dry_run=args.dry_run,
        download_images=args.download_images,
        reuse_existing_images=args.reuse_existing_images,
    )
    print(
        json.dumps(
            {
                "dataset": "eva",
                "record_count": result.record_count,
                "vote_count": result.vote_count,
                "scores_path": (
                    result.scores_path.as_posix() if result.scores_path else None
                ),
                "extracted_images": result.extracted_images,
                "dry_run": result.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
