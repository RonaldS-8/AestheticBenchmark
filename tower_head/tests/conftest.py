"""Shared fixtures: a tiny base-model snapshot and a CPU job built from it.

The snapshot looks like a Qwen3.5 checkpoint (full config, image-processor
config, sharded safetensors with an index) but its vision tower has 4 blocks
of width 32, so the whole job runs on CPU in seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from qv_head.config import JobConfig
from qv_head.weights import Fetch, extract_tower
from safetensors.torch import save_file
from tests.helpers import make_rows, write_dataset
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

TINY_VISION = {
    "depth": 4,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "out_hidden_size": 48,
    "num_position_embeddings": 64,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
# Two tapped blocks of width 32 plus the merger's 48.
TINY_FEATURE_DIM = 32 + 32 + 48


def write_snapshot(root: Path, *, extra_dtype: torch.dtype | None = None) -> Path:
    """A fake base-model snapshot; the tower sits in shard 2 with one LM tensor."""

    root.mkdir(parents=True)
    Qwen3_5Config(vision_config=TINY_VISION).to_json_file(str(root / "config.json"))
    Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ).save_pretrained(root)
    torch.manual_seed(0)
    tower = Qwen3_5VisionModel(Qwen3_5VisionConfig(**TINY_VISION))
    state = {f"model.visual.{k}": v.contiguous() for k, v in tower.state_dict().items()}
    if extra_dtype is not None:
        first = next(iter(state))
        state[first] = state[first].to(extra_dtype)
    shards = {
        "model.safetensors-00001-of-00002.safetensors": {
            "model.language_model.embed_tokens.weight": torch.zeros(4, 4)
        },
        "model.safetensors-00002-of-00002.safetensors": state,
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(root / name))
        weight_map |= {key: name for key in tensors}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return root


def snapshot_fetch(root: Path) -> Fetch:
    """Fetch files from a snapshot directory, like ``hf_hub_download``."""

    return lambda name: root / name


def tiny_config(**train: object) -> JobConfig:
    """CPU fp32 recipe on the tiny tower: batch 4, three epochs."""

    return JobConfig.model_validate(
        {
            "schema_version": "1.0",
            "recipe": "tiny-head",
            "tower": {
                "taps": [1, 3],
                "min_pixels": 1024,
                "max_pixels": 16384,
                "precision": "fp32",
            },
            "head": {"hidden_dim": 8, "dropout": 0.0},
            "loss": {"mse_weight": 1.0, "plcc_weight": 1.0, "rank_weight": 0.3},
            "train": {
                "batch_size": 4,
                "epochs": 3,
                "warmup_ratio": 0.1,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
            | train,
            "runtime": {"device": "cpu"},
        }
    )


@dataclass(frozen=True)
class TinyJob:
    """Paths of a laid-out job and a recipe that runs it in seconds."""

    data_root: Path
    models_dir: Path
    runs_dir: Path
    config: JobConfig


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``models/`` with the tiny tower cut out of a fake snapshot (built once)."""

    root = tmp_path_factory.mktemp("tower")
    snapshot = write_snapshot(root / "snapshot")
    models_dir = root / "models"
    extract_tower(snapshot_fetch(snapshot), models_dir, revision="0" * 40)
    return models_dir


@pytest.fixture
def tiny_job(tmp_path: Path, tiny_models: Path) -> TinyJob:
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_dataset(data_root, make_rows(12, 8, 4))
    return TinyJob(data_root, tiny_models, tmp_path / "runs", tiny_config())
