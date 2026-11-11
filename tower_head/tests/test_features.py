from __future__ import annotations

import pytest
import torch
from qv_head import features as features_module
from qv_head.data import load_dataset
from qv_head.features import (
    FEATURES_FILE,
    RECORD_FILE,
    FeatureError,
    extract_features,
    load_features,
)
from qv_head.records import FeatureSet
from tests.helpers import NOW, make_rows, write_dataset

from tests.conftest import TINY_FEATURE_DIM, TinyJob


def _extract(
    job: TinyJob, run_id: str = "tiny-features", **kwargs: object
) -> FeatureSet:
    options: dict[str, object] = {
        "data_root": job.data_root,
        "models_dir": job.models_dir,
        "runs_dir": job.runs_dir,
        "run_id": run_id,
        "limit": None,
        "now": NOW,
        "log": lambda _message: None,
    }
    return extract_features(job.config, **{**options, **kwargs})  # type: ignore[arg-type]


def test_extract_caches_train_and_val_only(
    tiny_job: TinyJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features_module, "LOG_EVERY", 5)
    messages: list[str] = []

    record = _extract(tiny_job, log=messages.append)

    assert (record.n_train, record.n_val) == (12, 8)
    assert record.feature_dim == TINY_FEATURE_DIM
    assert record.layer_names[-1] == "merger"
    assert record.tower == tiny_job.config.tower
    assert any("[特征] 5/20" in m for m in messages)
    assert any("[特征] 20/20" in m for m in messages)
    saved = FeatureSet.model_validate_json(
        (tiny_job.runs_dir / "tiny-features" / RECORD_FILE).read_text()
    )
    assert saved == record
    cached = load_features(
        tiny_job.runs_dir, "tiny-features", load_dataset(tiny_job.data_root)
    )
    assert cached.train.shape == (12, TINY_FEATURE_DIM)
    assert cached.val.shape == (8, TINY_FEATURE_DIM)
    assert all(i.split == "val" for i in cached.val_items)


def test_extract_with_limit_and_refuses_existing(tiny_job: TinyJob) -> None:
    record = _extract(tiny_job, "tiny-small", limit=3)

    assert (record.n_train, record.n_val, record.limited) == (3, 3, True)
    cached = load_features(
        tiny_job.runs_dir, "tiny-small", load_dataset(tiny_job.data_root)
    )
    assert cached.train.shape[0] == 3
    with pytest.raises(FeatureError, match="已存在"):
        _extract(tiny_job, "tiny-small")


def test_load_rejects_changed_cache_or_data(tiny_job: TinyJob) -> None:
    _extract(tiny_job)
    dataset = load_dataset(tiny_job.data_root)
    path = tiny_job.runs_dir / "tiny-features" / FEATURES_FILE
    saved = torch.load(path, weights_only=True)

    with pytest.raises(FeatureError, match="先运行"):
        load_features(tiny_job.runs_dir, "missing", dataset)
    other = tiny_job.data_root.parent / "other"
    other.mkdir()
    write_dataset(other, make_rows(13, 8, 4))
    with pytest.raises(FeatureError, match="数据"):
        load_features(tiny_job.runs_dir, "tiny-features", load_dataset(other))
    saved["features"] = saved["features"] + 1
    torch.save(saved, path)
    with pytest.raises(FeatureError, match="哈希"):
        load_features(tiny_job.runs_dir, "tiny-features", dataset)


def test_load_rejects_wrong_order_and_shape(tiny_job: TinyJob) -> None:
    record = _extract(tiny_job)
    dataset = load_dataset(tiny_job.data_root)
    path = tiny_job.runs_dir / "tiny-features" / FEATURES_FILE
    record_path = tiny_job.runs_dir / "tiny-features" / RECORD_FILE
    saved = torch.load(path, weights_only=True)

    def rewrite(content: dict[str, object]) -> None:
        torch.save(content, path)
        from qv_head.data import sha256_file

        updated = record.model_copy(
            update={
                "file": record.file.model_copy(update={"sha256": sha256_file(path)})
            }
        )
        record_path.write_text(updated.model_dump_json(), encoding="utf-8")

    rewrite({**saved, "image_ids": list(reversed(saved["image_ids"]))})
    with pytest.raises(FeatureError, match="顺序"):
        load_features(tiny_job.runs_dir, "tiny-features", dataset)
    rewrite({**saved, "features": saved["features"][:, :5]})
    with pytest.raises(FeatureError, match="形状"):
        load_features(tiny_job.runs_dir, "tiny-features", dataset)


def test_feature_record_checks_layers() -> None:
    base = {
        "schema_version": "1.0",
        "run_id": "f",
        "created_at": NOW,
        "items_sha256": "a" * 64,
        "tower_sha256": "b" * 64,
        "tower": {"taps": [0], "min_pixels": 1, "max_pixels": 2, "precision": "fp32"},
        "layer_names": ("block1", "merger"),
        "layer_dims": (4, 6),
        "n_train": 1,
        "n_val": 1,
        "limited": False,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 1.0,
        "file": {"path": "f/features.pt", "sha256": "c" * 64},
    }
    assert FeatureSet.model_validate(base).feature_dim == 10
    with pytest.raises(ValueError, match="数量"):
        FeatureSet.model_validate(base | {"layer_dims": (4, 6, 1)})
    with pytest.raises(ValueError, match="merger"):
        FeatureSet.model_validate(base | {"layer_names": ("merger", "block1")})
