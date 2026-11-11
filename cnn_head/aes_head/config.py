"""Fine-tuning configuration (TOML).

One file describes one training recipe. The seed and the backbone learning
rate are given per run on the command line, so a sweep is several runs of the
same file; ``selection.py`` refuses to compare runs whose recipes differ.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from aes_head.schema import Record, RunName, SchemaVersion


class HeadSettings(Record):
    """Two MLP heads on LayerNorm-ed multi-level pooled features."""

    hidden_dim: int = Field(ge=8, le=4096)
    dropout: float = Field(ge=0, lt=1)


class LossSettings(Record):
    """Per-score loss on z-scored targets: MSE + (1 - PLCC)/2 + pairwise rank."""

    mse_weight: float = Field(ge=0)
    plcc_weight: float = Field(ge=0)
    rank_weight: float = Field(ge=0)


class TrainSettings(Record):
    """Stage 1 trains the heads on a frozen backbone; stage 2 trains everything.

    In stage 2 the heads use ``backbone_lr * head_lr_multiplier``; the backbone
    learning rate itself comes from the command line.
    """

    batch_size: int = Field(ge=2, le=256)
    head_epochs: int = Field(ge=0)
    head_lr: float = Field(gt=0)
    finetune_epochs: int = Field(ge=1)
    head_lr_multiplier: float = Field(ge=1)
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    hflip: bool


class RuntimeSettings(Record):
    """Where to run; bf16 autocast needs CUDA."""

    device: str = Field(min_length=1)
    precision: Literal["bf16", "fp32"]
    num_workers: int = Field(ge=0)


class FinetuneConfig(Record):
    """A full fine-tuning recipe for one backbone."""

    schema_version: SchemaVersion
    recipe: RunName
    backbone: str = Field(min_length=1)
    head: HeadSettings
    loss: LossSettings
    train: TrainSettings
    runtime: RuntimeSettings


def load_config(path: Path) -> FinetuneConfig:
    """Read and validate a TOML recipe; unknown keys are errors."""

    with path.open("rb") as handle:
        return FinetuneConfig.model_validate(tomllib.load(handle))


def apply_overrides(
    config: FinetuneConfig, overrides: dict[str, dict[str, object]]
) -> FinetuneConfig:
    """Replace fields of the ``train`` / ``runtime`` tables and re-validate."""

    data = config.model_dump()
    for table, fields in overrides.items():
        if table not in ("train", "runtime"):
            raise ValueError(f"只能覆盖 train 或 runtime 表，实际 {table!r}")
        unknown = set(fields) - set(data[table])
        if unknown:
            raise ValueError(f"{table} 表没有这些字段: {sorted(unknown)}")
        data[table] |= fields
    return FinetuneConfig.model_validate(data)
