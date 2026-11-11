"""Download backbone checkpoints and pin them in ``models/WEIGHTS.json``.

torchvision names each checkpoint ``<name>-<sha256 prefix>.pth``; the download
is checked against that prefix and the full SHA-256 is recorded. A file
already on disk is checked, not downloaded again.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from aes_head.backbones import get_spec
from aes_head.data import sha256_file
from aes_head.schema import WeightsFile, WeightsIndex

INDEX_FILE = "WEIGHTS.json"
Downloader = Callable[..., None]


class WeightsError(ValueError):
    """Raised when a checkpoint does not match its expected hash."""


def hash_prefix(url: str) -> str:
    """The SHA-256 prefix torchvision puts in a checkpoint file name."""

    match = re.search(r"-([0-9a-f]{8,})\.pth$", url)
    if match is None:
        raise WeightsError(f"权重地址里没有哈希前缀: {url}")
    return match.group(1)


def load_index(models_dir: Path) -> WeightsIndex:
    """Read ``WEIGHTS.json``; a missing file is an error."""

    path = models_dir / INDEX_FILE
    if not path.is_file():
        raise WeightsError(f"找不到 {path}，先运行 download_weights.py")
    return WeightsIndex.model_validate_json(path.read_text(encoding="utf-8"))


def download_weights(
    models_dir: Path, backbones: Sequence[str], downloader: Downloader
) -> WeightsIndex:
    """Fetch missing checkpoints, verify all, and write the index."""

    models_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, WeightsFile] = {}
    if (models_dir / INDEX_FILE).is_file():
        entries = {entry.backbone: entry for entry in load_index(models_dir).files}
    for name in backbones:
        url = get_spec(name).weights_url
        target = models_dir / url.rsplit("/", 1)[1]
        prefix = hash_prefix(url)
        if not target.is_file():
            downloader(url, str(target), hash_prefix=prefix, progress=True)
        digest = sha256_file(target)
        if not digest.startswith(prefix):
            raise WeightsError(
                f"{target} 的哈希不以 {prefix} 开头，确认后手动删除再下载"
            )
        entry = WeightsFile(
            backbone=name,
            url=url,
            path=target.name,
            sha256=digest,
            size_bytes=target.stat().st_size,
        )
        old = entries.get(name)
        if old is not None and old != entry:
            raise WeightsError(f"{name} 的权重与 WEIGHTS.json 里记录的不同")
        entries[name] = entry
    index = WeightsIndex(schema_version="1.0", files=tuple(entries.values()))
    (models_dir / INDEX_FILE).write_text(
        index.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return index
