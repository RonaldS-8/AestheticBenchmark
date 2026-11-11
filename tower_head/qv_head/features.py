"""Extract tower features of train and val once and cache them.

``runs/<run_id>/features.pt`` holds the image ids and a ``[N, D]`` fp32
tensor, train rows first; ``features.json`` describes and pins it. The eval
split is never cached: it is scored end to end, once, by ``evaluation.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from pydantic import TypeAdapter

from qv_head.config import JobConfig
from qv_head.console import echo
from qv_head.data import Dataset, load_dataset, load_rgb, sha256_file
from qv_head.metrics import percentile
from qv_head.records import FeatureSet, FileRef
from qv_head.schema import HeadItem, RunName
from qv_head.tower import TowerFeatures, load_tower, sync

FEATURES_FILE = "features.pt"
RECORD_FILE = "features.json"
LOG_EVERY = 200


class FeatureError(ValueError):
    """Raised for an existing output or a cache that does not match the data."""


def _extract(
    features: TowerFeatures,
    root: Path,
    items: tuple[HeadItem, ...],
    log: Callable[[str], None],
) -> tuple[torch.Tensor, list[float]]:
    rows: list[torch.Tensor] = []
    latency: list[float] = []
    started = time.perf_counter()
    for count, item in enumerate(items, 1):
        encoded = features.encode(load_rgb(root, item))
        sync(features.device)
        start = time.perf_counter()
        rows.append(features(encoded).cpu())
        sync(features.device)
        latency.append((time.perf_counter() - start) * 1000)
        if count % LOG_EVERY == 0 or count == len(items):
            minutes = (time.perf_counter() - started) / 60
            log(
                f"[特征] {count}/{len(items)} 张，已用 {minutes:.1f} 分，"
                f"约剩 {minutes / count * (len(items) - count):.1f} 分"
            )
    return torch.stack(rows), latency


def extract_features(
    config: JobConfig,
    *,
    data_root: Path,
    models_dir: Path,
    runs_dir: Path,
    run_id: str,
    limit: int | None,
    now: datetime,
    log: Callable[[str], None] = echo,
) -> FeatureSet:
    """Cache train and val features under ``runs/<run_id>/``; never overwrites."""

    TypeAdapter(RunName).validate_python(run_id)
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FeatureError(f"{run_dir} 已存在；确认后删除再重跑，或换 run_id")
    dataset = load_dataset(data_root)
    train, val = dataset.split("train"), dataset.split("val")
    if limit is not None:
        train, val = train[:limit], val[:limit]
    tower, tower_sha256 = load_tower(models_dir, config.tower, config.runtime.device)
    log(
        f"{run_id}: 抽 train {len(train)} 张 + val {len(val)} 张的特征，"
        f"层 {' '.join(tower.layer_names)}，"
        f"设备 {config.runtime.device} {config.tower.precision}"
    )
    matrix, latency = _extract(tower, data_root, train + val, log)
    run_dir.mkdir(parents=True)
    path = run_dir / FEATURES_FILE
    torch.save(
        {"image_ids": [item.image_id for item in train + val], "features": matrix},
        path,
    )
    record = FeatureSet(
        schema_version="1.0",
        run_id=run_id,
        created_at=now,
        items_sha256=dataset.info.items_sha256,
        tower_sha256=tower_sha256,
        tower=config.tower,
        layer_names=tower.layer_names,
        layer_dims=tower.layer_dims,
        n_train=len(train),
        n_val=len(val),
        limited=limit is not None,
        latency_p50_ms=percentile(latency, 50),
        latency_p95_ms=percentile(latency, 95),
        file=FileRef(path=f"{run_id}/{FEATURES_FILE}", sha256=sha256_file(path)),
    )
    (run_dir / RECORD_FILE).write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return record


@dataclass(frozen=True)
class CachedFeatures:
    """Verified cached features split back into train and val."""

    record: FeatureSet
    train_items: tuple[HeadItem, ...]
    train: torch.Tensor
    val_items: tuple[HeadItem, ...]
    val: torch.Tensor


def load_features(runs_dir: Path, run_id: str, dataset: Dataset) -> CachedFeatures:
    """Read a feature cache and check it against its record and the dataset."""

    record_path = runs_dir / run_id / RECORD_FILE
    if not record_path.is_file():
        raise FeatureError(f"找不到 {record_path}，先运行 job.py features")
    record = FeatureSet.model_validate_json(record_path.read_text(encoding="utf-8"))
    if record.items_sha256 != dataset.info.items_sha256:
        raise FeatureError("特征抽取时用的数据与当前 data/ 不同")
    path = runs_dir / record.file.path
    if sha256_file(path) != record.file.sha256:
        raise FeatureError(f"{path} 的哈希与 {RECORD_FILE} 不符")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    train = dataset.split("train")[: record.n_train]
    val = dataset.split("val")[: record.n_val]
    if saved["image_ids"] != [item.image_id for item in train + val]:
        raise FeatureError("特征缓存的图片顺序与数据集不一致")
    matrix = saved["features"]
    if tuple(matrix.shape) != (len(train) + len(val), record.feature_dim):
        raise FeatureError(f"特征形状 {tuple(matrix.shape)} 与记录不符")
    return CachedFeatures(
        record, train, matrix[: len(train)], val, matrix[len(train) :]
    )
