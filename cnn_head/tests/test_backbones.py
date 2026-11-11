from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from aes_head.backbones import (
    BACKBONES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BackboneError,
    MultiLevelExtractor,
    build_backbone,
    get_spec,
    load_backbone,
    to_tensor,
)
from aes_head.data import sha256_file
from PIL import Image


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(BackboneError, match="未知骨干"):
        get_spec("vgg16")


@pytest.mark.parametrize("name", BACKBONES)
def test_extractor_pools_every_tapped_stage(name: str) -> None:
    spec = get_spec(name)
    extractor = MultiLevelExtractor(spec, build_backbone(name))
    pixels = to_tensor(Image.new("RGB", (45, 33), (10, 120, 200)))

    with torch.inference_mode():
        levels = extractor(pixels)

    assert tuple(level.shape for level in levels) == tuple((1, d) for d in spec.dims)


def test_to_tensor_keeps_size_and_normalizes() -> None:
    pixels = to_tensor(Image.new("RGB", (5, 3), (255, 0, 255)))

    assert pixels.shape == (1, 3, 3, 5)
    expected = [
        (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (1 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    ]
    assert pixels[0, :, 0, 0].tolist() == pytest.approx(expected)


def test_load_backbone_checks_hash(tmp_path: Path) -> None:
    path = tmp_path / "eff.pth"
    torch.save(build_backbone("efficientnet_b0").state_dict(), path)

    model = load_backbone("efficientnet_b0", path, sha256_file(path))

    assert isinstance(model, torch.nn.Module)
    with pytest.raises(BackboneError, match="哈希"):
        load_backbone("efficientnet_b0", path, "0" * 64)
    with pytest.raises(BackboneError, match="不存在"):
        load_backbone("efficientnet_b0", tmp_path / "missing.pth", "0" * 64)


def test_extractor_rejects_missing_layer_or_wrong_dims() -> None:
    spec = get_spec("efficientnet_b0")
    model = build_backbone("efficientnet_b0")
    pixels = to_tensor(Image.new("RGB", (40, 32)))

    with pytest.raises(BackboneError, match="没有这些层"):
        MultiLevelExtractor(replace(spec, taps=("features.99",)), model)
    extractor = MultiLevelExtractor(replace(spec, dims=(1, 2, 3, 4)), model)
    with pytest.raises(BackboneError, match="通道数"), torch.inference_mode():
        extractor(pixels)
