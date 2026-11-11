"""Copy the EVA split and images from the main repository into ``data/``.

Run once, inside the main repository checkout; afterwards this folder needs
nothing from the main repository. ``--dry-run`` checks every file and hash
and prints what would be written, without writing.

Example::

    python cnn_head/prepare_data.py --dry-run
    python cnn_head/prepare_data.py
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

from aes_head.prepare import plan_copy, write_copy  # noqa: E402

DEFAULT_MANIFEST = "manifests/eva/sft/v1/manifest.json"


def main(argv: list[str] | None = None) -> int:
    """Plan the copy, then write it unless ``--dry-run``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root", type=Path, default=HERE.parent / "lora" / "data"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plan = plan_copy(args.source_root, args.manifest, now=datetime.now().astimezone())
    summary = {
        "dry_run": args.dry_run,
        "output": str(args.output),
        "counts": plan.info.counts,
        "image_mb": round(plan.image_bytes / 2**20, 1),
        "eval_fingerprint": plan.info.eval_fingerprint,
        "items_sha256": plan.info.items_sha256,
    }
    if not args.dry_run:
        write_copy(plan, args.source_root, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
