from __future__ import annotations

from pathlib import Path

import pytest
import torch
from qv_head.config import load_config, with_device
from qv_head.data import sha256_file
from qv_head.head import (
    FeatureHead,
    HeadError,
    build_head,
    load_head,
    save_head,
)
from qv_head.schema import EVA_DIMENSIONS

from tests.conftest import tiny_config

HERE = Path(__file__).resolve().parents[1]


def _head() -> FeatureHead:
    torch.manual_seed(0)
    x = torch.randn(10, 6)
    x[:, 2] = 3.0  # constant on train
    y = torch.rand(10, 2) * 4 + 1
    return build_head(tiny_config(), x, y)


def test_forward_standardizes_and_scores() -> None:
    head = _head()

    z = head(torch.randn(5, 6))

    assert z.shape == (5, len(EVA_DIMENSIONS))
    assert head.feature_std[2] == 1.0
    scores = head.to_scores(torch.zeros(1, 2))
    assert torch.allclose(scores[0], head.target_mean)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(HeadError, match="形状"):
        FeatureHead(
            4,
            hidden_dim=8,
            dropout=0.0,
            feature_mean=torch.zeros(3),
            feature_std=torch.ones(4),
            target_mean=[0.0, 0.0],
            target_std=[1.0, 1.0],
        )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    head = _head()
    path = tmp_path / "best.pt"
    save_head(path, head, tiny_config(), "d" * 64)

    loaded, config, tower = load_head(path, sha256_file(path))

    assert config == tiny_config() and tower == "d" * 64
    x = torch.randn(3, 6)
    head.eval()
    loaded.eval()
    assert torch.equal(head(x), loaded(x))


def test_load_rejects_bad_files(tmp_path: Path) -> None:
    head = _head()
    path = tmp_path / "best.pt"
    save_head(path, head, tiny_config(), "d" * 64)

    with pytest.raises(HeadError, match="哈希"):
        load_head(path, "0" * 64)
    with pytest.raises(HeadError, match="不存在"):
        load_head(tmp_path / "missing.pt", "0" * 64)
    saved = torch.load(path, weights_only=True)
    torch.save({**saved, "format": 99}, path)
    with pytest.raises(HeadError, match="格式"):
        load_head(path, sha256_file(path))


def test_shipped_config_is_valid_and_overridable() -> None:
    config = load_config(HERE / "configs" / "qwen35-4b-tower-head.toml")

    assert config.tower.taps == (5, 11, 17, 23)
    assert (config.tower.min_pixels, config.tower.max_pixels) == (65536, 1048576)
    cpu = with_device(config, "cpu", "fp32")
    assert (cpu.runtime.device, cpu.tower.precision) == ("cpu", "fp32")
    assert cpu.train == config.train


def test_tower_settings_are_checked() -> None:
    tower = tiny_config().tower.model_dump()

    with pytest.raises(ValueError, match="taps"):
        type(tiny_config().tower).model_validate(tower | {"taps": [3, 1]})
    with pytest.raises(ValueError, match="min_pixels"):
        type(tiny_config().tower).model_validate(tower | {"min_pixels": 10**6})
