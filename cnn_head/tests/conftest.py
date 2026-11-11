"""Shared fixture: a tiny fine-tuning job on CPU with a two-stage toy CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from aes_head import backbones
from aes_head.backbones import BackboneSpec
from aes_head.config import FinetuneConfig
from aes_head.data import sha256_file
from aes_head.schema import WeightsFile, WeightsIndex
from tests.helpers import make_rows, tiny_cnn, write_dataset

TINY = BackboneSpec(
    name="tiny_cnn",
    build=tiny_cnn,
    weights_url="https://example.test/tiny_cnn-00000000.pth",
    taps=("1", "3"),
    dims=(8, 16),
)


@dataclass(frozen=True)
class TinyJob:
    """Paths of a laid-out job and a recipe that trains it in seconds."""

    data_root: Path
    models_dir: Path
    runs_dir: Path
    config: FinetuneConfig


def write_tiny_weights(models_dir: Path) -> None:
    """Save a seeded toy backbone and index it like download_weights.py does."""

    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "tiny.pth"
    torch.manual_seed(0)
    torch.save(tiny_cnn().state_dict(), path)
    entry = WeightsFile(
        backbone=TINY.name,
        url=TINY.weights_url,
        path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )
    index = WeightsIndex(schema_version="1.0", files=(entry,))
    (models_dir / "WEIGHTS.json").write_text(index.model_dump_json(), encoding="utf-8")


def tiny_config(**train: object) -> FinetuneConfig:
    """CPU fp32 recipe: batch 4, one epoch per stage."""

    return FinetuneConfig.model_validate(
        {
            "schema_version": "1.0",
            "recipe": "tiny-ft",
            "backbone": TINY.name,
            "head": {"hidden_dim": 8, "dropout": 0.0},
            "loss": {"mse_weight": 1.0, "plcc_weight": 1.0, "rank_weight": 0.3},
            "train": {
                "batch_size": 4,
                "head_epochs": 1,
                "head_lr": 1e-2,
                "finetune_epochs": 1,
                "head_lr_multiplier": 10.0,
                "warmup_ratio": 0.1,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "hflip": True,
            }
            | train,
            "runtime": {"device": "cpu", "precision": "fp32", "num_workers": 0},
        }
    )


@pytest.fixture
def tiny_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TinyJob:
    monkeypatch.setitem(backbones.SPECS, TINY.name, TINY)
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_dataset(data_root, make_rows(12, 6, 4))
    write_tiny_weights(tmp_path / "models")
    return TinyJob(data_root, tmp_path / "models", tmp_path / "runs", tiny_config())
