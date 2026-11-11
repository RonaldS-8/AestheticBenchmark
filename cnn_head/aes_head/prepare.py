"""Copy the EVA split and images out of the main repository.

Reads the main repository's EVA SFT manifest, split file and sample JSONL as
plain JSON (``lora_sft`` is not imported), keeps only what the head needs
(id, split, image path and hash, vote summaries) and copies every image. The
output directory is written once and never overwritten.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from aes_head.data import (
    DATASET_FILE,
    ids_fingerprint,
    items_jsonl_bytes,
    sha256_bytes,
    sha256_file,
)
from aes_head.schema import (
    SPLITS,
    DatasetInfo,
    HeadItem,
    ImageId,
    RelativePath,
    ScoreSummary,
    Sha256,
    SourceRef,
    Split,
)

ITEMS_FILE = "items.jsonl"


class PrepareError(ValueError):
    """Raised when the source files disagree or the output already exists."""


class _Foreign(BaseModel):
    """Reads a main-repository record, ignoring fields the head does not use."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class _SourceManifest(_Foreign):
    dataset: str
    split_path: RelativePath
    train_path: RelativePath
    val_path: RelativePath
    eval_path: RelativePath
    train_count: int
    val_count: int
    eval_count: int
    eval_fingerprint: Sha256


class _SourceSplit(_Foreign):
    split_version: str
    train: tuple[ImageId, ...]
    val: tuple[ImageId, ...]
    eval: tuple[ImageId, ...]


class _SourceLabels(_Foreign):
    aesthetic: ScoreSummary
    technical: ScoreSummary


class _SourceSample(_Foreign):
    image_id: ImageId
    split: Split
    image_path: RelativePath
    image_sha256: Sha256
    image_decode_warnings: tuple[str, ...]
    source_labels: _SourceLabels


@dataclass(frozen=True)
class PreparePlan:
    """What would be written: the items, their description and image bytes."""

    items: tuple[HeadItem, ...]
    info: DatasetInfo
    items_bytes: bytes
    image_bytes: int


def _read_split(
    source_root: Path, path: str, name: Split, expected: tuple[str, ...], count: int
) -> list[HeadItem]:
    lines = (source_root / path).read_text(encoding="utf-8").splitlines()
    items: list[HeadItem] = []
    for line in lines:
        sample = _SourceSample.model_validate_json(line)
        if sample.split != name:
            raise PrepareError(
                f"{path} 里有 {sample.split} 划分的样本 {sample.image_id}"
            )
        items.append(
            HeadItem(
                image_id=sample.image_id,
                split=sample.split,
                image_path=sample.image_path,
                image_sha256=sample.image_sha256,
                image_decode_warnings=sample.image_decode_warnings,
                aesthetic=sample.source_labels.aesthetic,
                technical=sample.source_labels.technical,
            )
        )
    ids = [item.image_id for item in items]
    if len(ids) != count or sorted(ids) != sorted(expected):
        raise PrepareError(f"{name} 样本与划分文件或清单数量不一致")
    return items


def plan_copy(source_root: Path, manifest: str, *, now: datetime) -> PreparePlan:
    """Read and cross-check the source files; verify every image hash."""

    manifest_bytes = (source_root / manifest).read_bytes()
    source = _SourceManifest.model_validate_json(manifest_bytes)
    if source.dataset != "eva":
        raise PrepareError(f"只支持 EVA 清单，实际 {source.dataset!r}")
    split = _SourceSplit.model_validate(
        json.loads((source_root / source.split_path).read_text(encoding="utf-8"))
    )
    items: list[HeadItem] = []
    for name in SPLITS:
        items += _read_split(
            source_root,
            getattr(source, f"{name}_path"),
            name,
            getattr(split, name),
            getattr(source, f"{name}_count"),
        )
    ids = [item.image_id for item in items]
    if len(set(ids)) != len(ids):
        raise PrepareError("不同划分之间有重复的 image_id")
    eval_ids = [item.image_id for item in items if item.split == "eval"]
    if ids_fingerprint(eval_ids) != source.eval_fingerprint:
        raise PrepareError("重算的 eval 指纹与源清单不符")
    image_bytes = 0
    for item in items:
        path = source_root / item.image_path
        if not path.is_file():
            raise PrepareError(f"源图片不存在: {path}")
        if sha256_file(path) != item.image_sha256:
            raise PrepareError(f"源图片哈希与清单不符: {path}")
        image_bytes += path.stat().st_size
    items_bytes = items_jsonl_bytes(items)
    info = DatasetInfo(
        schema_version="1.0",
        dataset="eva",
        created_at=now,
        source_manifest=SourceRef(path=manifest, sha256=sha256_bytes(manifest_bytes)),
        split_version=split.split_version,
        items_path=ITEMS_FILE,
        items_sha256=sha256_bytes(items_bytes),
        counts={name: sum(i.split == name for i in items) for name in SPLITS},
        eval_fingerprint=source.eval_fingerprint,
    )
    return PreparePlan(tuple(items), info, items_bytes, image_bytes)


def write_copy(plan: PreparePlan, source_root: Path, out_root: Path) -> None:
    """Copy images and write the items and ``dataset.json`` (last)."""

    if out_root.exists() and any(out_root.iterdir()):
        raise PrepareError(f"{out_root} 已存在且不为空，不覆盖；确认后手动删除再重跑")
    for item in plan.items:
        target = out_root / item.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / item.image_path, target)
        if sha256_file(target) != item.image_sha256:
            raise PrepareError(f"复制后的图片哈希不符: {target}")
    (out_root / ITEMS_FILE).write_bytes(plan.items_bytes)
    (out_root / DATASET_FILE).write_text(
        plan.info.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
