"""Download the ImageNet checkpoints of the backbones into ``models/``.

Example::

    python download_weights.py
    python download_weights.py --backbones convnext_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aes_head.backbones import BACKBONES  # noqa: E402
from aes_head.weights import download_weights  # noqa: E402
from torch.hub import download_url_to_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Download missing checkpoints and print the index."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    parser.add_argument("--models-dir", type=Path, default=HERE / "models")
    args = parser.parse_args(argv)

    index = download_weights(args.models_dir, args.backbones, download_url_to_file)
    for entry in index.files:
        print(f"{entry.backbone}: {entry.path} {entry.size_bytes / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
