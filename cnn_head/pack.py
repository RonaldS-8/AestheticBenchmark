"""Pack the fine-tuning job into one tar for the cloud (run locally).

Example::

    python pack.py --config configs/convnext-t-ft.toml --dry-run
    python pack.py --config configs/convnext-t-ft.toml
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
from aes_head.packing import git_version, plan_pack, write_tar  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Verify everything, then write the tar unless ``--dry-run``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    recipe = load_config(args.config).recipe
    output = args.output or HERE / "dist" / f"aesthetic-head-{recipe}.tar"
    entries = plan_pack(HERE, args.config)
    commit, dirty = git_version(HERE)
    size_mb = sum(entry.size() for entry in entries) / 2**20
    print(f"{len(entries)} 个文件，{size_mb:.0f} MB，commit {commit[:10]}")
    if dirty:
        print("注意：cnn_head/ 有未提交的改动，BUILD_INFO 会记下 git_dirty")
    if not args.dry_run:
        write_tar(
            output, entries, commit=commit, dirty=dirty, now=datetime.now().astimezone()
        )
        print(f"已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
