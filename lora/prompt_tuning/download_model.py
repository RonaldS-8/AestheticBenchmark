"""Download the job's pinned model weights into ``models/`` (run once).

Run from the job root. Safe to re-run: an interrupted download resumes, and
a finished one is only re-checked. Choose the source explicitly::

    HF_ENDPOINT=https://hf-mirror.com python download_model.py --source hf
    python download_model.py --source modelscope

Either way every file must match the SHA-256 recorded in ``data/job.json``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptune.download import download_model  # noqa: E402
from ptune.schema import JobConfig  # noqa: E402

SHANGHAI = timezone(timedelta(hours=8))


def main(argv: list[str] | None = None) -> int:
    """Download (or re-check) the weights and print where they are."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("hf", "modelscope"), required=True)
    parser.add_argument("--job", type=Path, default=Path("data/job.json"))
    args = parser.parse_args(argv)
    config = JobConfig.model_validate_json(args.job.read_text(encoding="utf-8"))
    folder = download_model(config, Path.cwd(), args.source, now=datetime.now(SHANGHAI))
    print(f"权重就绪：{folder}（{config.model.model_id}@{config.model.revision}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
