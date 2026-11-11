from __future__ import annotations

from pathlib import Path

import pytest
import torch
from aes_head.backbones import MultiLevelExtractor, build_backbone, get_spec
from aes_head.data import Dataset, load_dataset
from aes_head.features import (
    FeatureError,
    FeatureSet,
    extract_features,
    load_features,
    save_features,
)
from tests.helpers import make_rows, write_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    write_dataset(tmp_path, make_rows(4, 2, 1))
    return load_dataset(tmp_path)


def _extract(dataset: Dataset, splits: tuple[str, ...]) -> FeatureSet:
    spec = get_spec("efficientnet_b0")
    items = [i for i in dataset.items if i.split in splits]
    return extract_features(
        dataset.root,
        items,
        MultiLevelExtractor(spec, build_backbone(spec.name)),
        weights_sha256="c" * 64,
        items_sha256=dataset.info.items_sha256,
        device="cpu",
        log=lambda _message: None,
    )


def test_extract_features_one_row_per_image(dataset: Dataset) -> None:
    features = _extract(dataset, ("train", "val"))

    assert len(features.image_ids) == 6
    assert [level.shape for level in features.levels] == [
        (6, 24),
        (6, 40),
        (6, 112),
        (6, 1280),
    ]
    assert features.levels[0].dtype == torch.float64
    assert features.matrix("all").shape == (6, 1456)
    assert features.matrix("last").shape == (6, 1280)
    assert len(features.latency_ms) == 6


def test_extract_features_refuses_eval(dataset: Dataset) -> None:
    with pytest.raises(FeatureError, match="eval"):
        _extract(dataset, ("train", "eval"))


def test_matrix_rejects_unknown_layer_set(dataset: Dataset) -> None:
    with pytest.raises(FeatureError, match="layers"):
        _extract(dataset, ("val",)).matrix("middle")


def test_cache_round_trip_and_checks(dataset: Dataset, tmp_path: Path) -> None:
    features = _extract(dataset, ("train",))
    path = tmp_path / "features.pt"
    request = {
        "backbone": "efficientnet_b0",
        "weights_sha256": "c" * 64,
        "items_sha256": dataset.info.items_sha256,
        "image_ids": features.image_ids,
    }
    save_features(path, features)

    loaded = load_features(path, **request)  # type: ignore[arg-type]

    assert loaded.image_ids == features.image_ids
    assert all(
        torch.equal(a, b) for a, b in zip(loaded.levels, features.levels, strict=True)
    )
    with pytest.raises(FeatureError, match="不覆盖"):
        save_features(path, features)
    with pytest.raises(FeatureError, match="weights_sha256"):
        load_features(path, **{**request, "weights_sha256": "d" * 64})  # type: ignore[arg-type]


def test_cache_with_wrong_row_count_is_rejected(
    dataset: Dataset, tmp_path: Path
) -> None:
    features = _extract(dataset, ("train",))
    short = FeatureSet(
        backbone=features.backbone,
        weights_sha256=features.weights_sha256,
        items_sha256=features.items_sha256,
        image_ids=features.image_ids,
        levels=tuple(level[:2] for level in features.levels),
        latency_ms=features.latency_ms,
    )
    path = tmp_path / "short.pt"
    save_features(path, short)

    with pytest.raises(FeatureError, match="行数"):
        load_features(
            path,
            backbone=short.backbone,
            weights_sha256=short.weights_sha256,
            items_sha256=short.items_sha256,
            image_ids=short.image_ids,
        )
