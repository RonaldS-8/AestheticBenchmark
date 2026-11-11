"""Choose the best run of a sweep on val and write ``runs/<prefix>-selection.json``.

Example::

    python select_run.py --prefix convnext-t-ft
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aes_head.selection import SelectionError, select_sweep  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Select, print the table and write the selection file once."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = parser.parse_args(argv)

    selection = select_sweep(
        args.runs_dir, args.prefix, now=datetime.now().astimezone()
    )
    output = args.runs_dir / f"{args.prefix}-selection.json"
    if output.exists():
        raise SelectionError(f"{output} 已存在，不覆盖")
    output.write_text(selection.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for group in selection.groups:
        values = " ".join(f"{v:.3f}" for v in group.best_val_srcc_mean)
        print(f"backbone_lr {group.backbone_lr:g}: 均值 {group.mean:.3f}（{values}）")
    print(f"选中 {selection.chosen_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
