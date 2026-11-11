"""Load the copied EVA dataset and decode its images.

Everything is checked on load: the items file hash, split counts, the eval
fingerprint and, for every image read, its SHA-256. Images are decoded the
way the main repository does it (Pillow, then ``convert("RGB")``, no EXIF
rotation), at their stored size.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from qv_head.schema import SPLITS, DatasetInfo, HeadItem, Split

DATASET_FILE = "dataset.json"


class DatasetError(ValueError):
    """Raised when the copied dataset is missing or inconsistent."""


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_fingerprint(image_ids: Iterable[str]) -> str:
    """Order-independent SHA-256 of image ids (same as the main repository)."""

    return sha256_bytes("\n".join(sorted(image_ids)).encode("utf-8"))


def items_jsonl_bytes(items: Sequence[HeadItem]) -> bytes:
    """Serialize items as UTF-8 JSONL, one record per line."""

    return "".join(item.model_dump_json() + "\n" for item in items).encode("utf-8")


@dataclass(frozen=True)
class Dataset:
    """The verified dataset: its root, description and items."""

    root: Path
    info: DatasetInfo
    items: tuple[HeadItem, ...]

    def split(self, name: Split) -> tuple[HeadItem, ...]:
        """Items of one split, in file order."""

        return tuple(item for item in self.items if item.split == name)


def load_dataset(root: Path) -> Dataset:
    """Read and verify ``dataset.json`` and the items file under ``root``."""

    info_path = root / DATASET_FILE
    if not info_path.is_file():
        raise DatasetError(f"找不到 {info_path}，先运行 prepare_data.py")
    info = DatasetInfo.model_validate_json(info_path.read_text(encoding="utf-8"))
    raw = (root / info.items_path).read_bytes()
    if sha256_bytes(raw) != info.items_sha256:
        raise DatasetError(f"{info.items_path} 的哈希与 dataset.json 不符")
    items = tuple(
        HeadItem.model_validate_json(line) for line in raw.decode("utf-8").splitlines()
    )
    ids = [item.image_id for item in items]
    if len(set(ids)) != len(ids):
        raise DatasetError("items 里有重复的 image_id")
    counts = {name: sum(item.split == name for item in items) for name in SPLITS}
    if counts != info.counts:
        raise DatasetError(f"各划分数量 {counts} 与 dataset.json {info.counts} 不符")
    eval_ids = [item.image_id for item in items if item.split == "eval"]
    if ids_fingerprint(eval_ids) != info.eval_fingerprint:
        raise DatasetError("eval 指纹与 dataset.json 不符")
    return Dataset(root, info, items)


def load_rgb(root: Path, item: HeadItem) -> Image.Image:
    """Decode an item's image as RGB after checking its SHA-256."""

    path = root / item.image_path
    if not path.is_file():
        raise DatasetError(f"图片不存在: {path}")
    data = path.read_bytes()
    if sha256_bytes(data) != item.image_sha256:
        raise DatasetError(f"图片哈希与清单不符: {path}")
    with Image.open(io.BytesIO(data)) as opened:
        return opened.convert("RGB")
