from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from qv_head.tower import TowerError, check_runtime, load_tower

from tests.conftest import TINY_FEATURE_DIM, tiny_config


def test_check_runtime_rejects_bf16_on_cpu() -> None:
    with pytest.raises(TowerError, match="fp32"):
        check_runtime("cpu", "bf16")
    if not torch.cuda.is_available():
        with pytest.raises(TowerError, match="CUDA"):
            check_runtime("cuda", "fp32")


def test_features_pool_tapped_blocks_and_merger(tiny_models: Path) -> None:
    tower, sha = load_tower(tiny_models, tiny_config().tower, "cpu")
    image = Image.new("RGB", (70, 40), (10, 200, 30))

    encoded = tower.encode(image)
    first = tower(encoded)
    again = tower(tower.encode(image))
    other = tower(tower.encode(Image.new("RGB", (70, 40), (200, 10, 90))))

    assert tower.layer_names == ("block2", "block4", "merger")
    assert sum(tower.layer_dims) == TINY_FEATURE_DIM == first.shape[0]
    assert first.dtype == torch.float32
    assert torch.equal(first, again)
    assert not torch.allclose(first, other)
    assert len(sha) == 64


def test_encode_uses_the_pixel_budget(tiny_models: Path) -> None:
    tower, _ = load_tower(tiny_models, tiny_config().tower, "cpu")

    small = tower.encode(Image.new("RGB", (20, 20)))
    large = tower.encode(Image.new("RGB", (400, 400)))

    # min 1024 pixels -> at least 32x32; max 16384 -> at most 128x128 (16 px patches)
    assert small["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert large["image_grid_thw"].tolist() == [[1, 8, 8]]


def test_taps_beyond_the_tower_are_rejected(tiny_models: Path) -> None:
    settings = tiny_config().tower.model_copy(update={"taps": (1, 4)})

    with pytest.raises(TowerError, match="超出"):
        load_tower(tiny_models, settings, "cpu")
