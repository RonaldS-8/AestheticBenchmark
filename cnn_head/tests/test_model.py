from __future__ import annotations

from pathlib import Path

import pytest
import torch
from aes_head.backbones import build_backbone
from aes_head.config import apply_overrides, load_config
from aes_head.model import HeadModel, strip_classifier
from tests.helpers import HERE, tiny_cnn
from torch import nn

from tests.conftest import TinyJob


def _model() -> HeadModel:
    torch.manual_seed(0)
    return HeadModel(
        "tiny_cnn",
        tiny_cnn(),
        hidden_dim=8,
        dropout=0.0,
        target_mean=[3.0, 2.5],
        target_std=[0.5, 0.25],
    )


def test_images_of_different_sizes_share_a_batch(tiny_job: TinyJob) -> None:
    model = _model()
    images = [torch.randn(3, 20, 30), torch.randn(3, 33, 17)]

    z = model(images)

    assert z.shape == (2, 2)
    assert model.to_scores(torch.zeros(1, 2)).tolist() == [[3.0, 2.5]]
    assert model.to_scores(torch.ones(1, 2)).tolist() == [[3.5, 2.75]]


def test_frozen_backbone_gets_no_gradient(tiny_job: TinyJob) -> None:
    model = _model()

    model([torch.randn(3, 16, 16), torch.randn(3, 16, 20)], train_backbone=False)[
        :, 0
    ].sum().backward()

    assert all(p.grad is None for p in model.backbone_parameters())
    assert any(p.grad is not None for p in model.head_parameters())


def test_strip_classifier_drops_unused_layers() -> None:
    model = build_backbone("convnext_tiny")

    strip_classifier(model)

    assert isinstance(model.classifier, nn.Identity)


def test_shipped_recipe_loads_and_overrides_are_checked() -> None:
    config = load_config(HERE / "configs" / "convnext-t-ft.toml")

    smoke = apply_overrides(
        config, {"runtime": {"device": "cpu", "precision": "fp32"}, "train": {}}
    )

    assert config.backbone == "convnext_tiny"
    assert smoke.runtime.device == "cpu" and smoke.train == config.train
    with pytest.raises(ValueError, match="只能覆盖"):
        apply_overrides(config, {"head": {"dropout": 0.5}})
    with pytest.raises(ValueError, match="没有这些字段"):
        apply_overrides(config, {"train": {"epochs": 3}})


def test_recipe_rejects_unknown_keys(tmp_path: Path) -> None:
    text = (HERE / "configs" / "convnext-t-ft.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(text + "\nextra = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extra"):
        load_config(path)
