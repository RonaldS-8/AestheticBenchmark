"""Builders for tiny datasets used across the tests."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

from aes_head.data import (
    DATASET_FILE,
    ids_fingerprint,
    items_jsonl_bytes,
    sha256_bytes,
    sha256_file,
)
from aes_head.schema import DatasetInfo, HeadItem, ScoreSummary, SourceRef
from PIL import Image
from torch import nn

HERE = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
# image id -> (split, aesthetic mean 0-10, technical mean 1-4)
Rows = dict[str, tuple[str, float, float]]


def make_rows(train: int, val: int, eval_: int) -> Rows:
    """Rows with spread-out labels so correlations are defined."""

    rows: Rows = {}
    for offset in range(train + val + eval_):
        split = "train" if offset < train else "val" if offset < train + val else "eval"
        aesthetic = 2 + (offset * 37 % 60) / 10
        technical = 1.5 + (offset * 13 % 20) / 10
        rows[f"photo_{offset + 1:06d}"] = (split, aesthetic, technical)
    return rows


def tiny_cnn() -> nn.Module:
    """Two conv stages; randomly initialized, its pooled features still vary.

    Random ImageNet-sized backbones in eval mode give near-constant deep
    features, so pipeline tests use this instead.
    """

    return nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        nn.ReLU(),
    )


def summary(mean: float, low: float, high: float) -> ScoreSummary:
    return ScoreSummary(
        mean=mean, standard_deviation=1.0, count=30, scale_min=low, scale_max=high
    )


def write_images(root: Path, rows: Rows) -> dict[str, str]:
    """Write one small JPEG per row; return image id -> relative path."""

    paths = {}
    for offset, image_id in enumerate(rows):
        relative = f"images/eva/{int(image_id[6:])}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        size = (40 + offset % 7, 32)
        noise = random.Random(offset).randbytes(size[0] * size[1] * 3)
        Image.frombytes("RGB", size, noise).save(path, "JPEG")
        paths[image_id] = relative
    return paths


def make_items(root: Path, rows: Rows) -> list[HeadItem]:
    paths = write_images(root, rows)
    return [
        HeadItem(
            image_id=image_id,
            split=split,  # type: ignore[arg-type]
            image_path=paths[image_id],
            image_sha256=sha256_file(root / paths[image_id]),
            image_decode_warnings=(),
            aesthetic=summary(aesthetic, 0, 10),
            technical=summary(technical, 1, 4),
        )
        for image_id, (split, aesthetic, technical) in rows.items()
    ]


def write_dataset(root: Path, rows: Rows) -> DatasetInfo:
    """Lay out ``data/`` the way prepare_data.py writes it."""

    items = make_items(root, rows)
    raw = items_jsonl_bytes(items)
    (root / "items.jsonl").write_bytes(raw)
    info = DatasetInfo(
        schema_version="1.0",
        dataset="eva",
        created_at=NOW,
        source_manifest=SourceRef(
            path="manifests/eva/sft/v1/manifest.json", sha256="a" * 64
        ),
        split_version="split-v1",
        items_path="items.jsonl",
        items_sha256=sha256_bytes(raw),
        counts={
            name: sum(item.split == name for item in items)
            for name in ("train", "val", "eval")
        },
        eval_fingerprint=ids_fingerprint(
            item.image_id for item in items if item.split == "eval"
        ),
    )
    (root / DATASET_FILE).write_text(info.model_dump_json(indent=2), encoding="utf-8")
    return info


def write_source(root: Path, rows: Rows) -> str:
    """Lay out a main-repository ``data/`` with an EVA SFT manifest.

    Sample rows carry extra fields (messages, prompt id) like the real ones.
    Returns the manifest path relative to ``root``.
    """

    items = make_items(root, rows)
    folder = "manifests/eva/sft/v1"
    (root / folder).mkdir(parents=True)
    by_split: dict[str, list[HeadItem]] = {"train": [], "val": [], "eval": []}
    for item in items:
        by_split[item.split].append(item)
    for name, group in by_split.items():
        lines = []
        for item in group:
            lines.append(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "image_id": item.image_id,
                        "split": item.split,
                        "prompt_id": "eva-dual-score@v1",
                        "image_path": item.image_path,
                        "image_sha256": item.image_sha256,
                        "image_decode_warnings": [],
                        "source_labels": {
                            "capture_id": "eva-test",
                            "aesthetic": item.aesthetic.model_dump(),
                            "technical": item.technical.model_dump(),
                        },
                        "messages": [],
                    }
                )
            )
        (root / folder / f"{name}.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
    split = {"schema_version": "1.0", "split_version": "split-v1", "seed": 42}
    split |= {name: [i.image_id for i in group] for name, group in by_split.items()}
    (root / folder / "split-v1.json").write_text(json.dumps(split), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset": "eva",
        "split_path": f"{folder}/split-v1.json",
        "train_path": f"{folder}/train.jsonl",
        "val_path": f"{folder}/val.jsonl",
        "eval_path": f"{folder}/eval.jsonl",
        "train_count": len(by_split["train"]),
        "val_count": len(by_split["val"]),
        "eval_count": len(by_split["eval"]),
        "eval_fingerprint": ids_fingerprint(i.image_id for i in by_split["eval"]),
        "record_count": len(items),
    }
    (root / folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return f"{folder}/manifest.json"
