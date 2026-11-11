"""Job configuration (TOML).

One file describes the tower features and the head recipe. The seed and the
head learning rate are given per run, so a sweep is several runs of the same
file; ``selection.py`` refuses to compare runs whose recipes differ.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from qv_head.schema import Record, RunName, SchemaVersion


class TowerSettings(Record):
    """Which tower outputs are pooled, and the image pixel budget.

    ``taps`` are 0-based block indices; each tapped block's output tokens and
    the merger's output tokens are mean-pooled and concatenated. The pixel
    budget is the LoRA config's ``[preprocess]``, so both see the same input.
    """

    taps: tuple[int, ...] = Field(min_length=1)
    min_pixels: int = Field(gt=0)
    max_pixels: int = Field(gt=0)
    precision: Literal["bf16", "fp32"]

    @model_validator(mode="after")
    def _check(self) -> TowerSettings:
        if list(self.taps) != sorted(set(self.taps)) or self.taps[0] < 0:
            raise ValueError("taps 必须是不重复、递增的非负块编号")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels 不能大于 max_pixels")
        return self


class HeadSettings(Record):
    """Two MLP heads on standardized pooled features."""

    hidden_dim: int = Field(ge=8, le=4096)
    dropout: float = Field(ge=0, lt=1)


class LossSettings(Record):
    """Per-score loss on z-scored targets: MSE + (1 - PLCC)/2 + pairwise rank."""

    mse_weight: float = Field(ge=0)
    plcc_weight: float = Field(ge=0)
    rank_weight: float = Field(ge=0)


class TrainSettings(Record):
    """Head training on cached features; the learning rate is per run."""

    batch_size: int = Field(ge=2, le=1024)
    epochs: int = Field(ge=1)
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)


class RuntimeSettings(Record):
    """Where to run; the tower in bf16 needs CUDA."""

    device: str = Field(min_length=1)


class JobConfig(Record):
    """Tower features plus the head recipe trained on them."""

    schema_version: SchemaVersion
    recipe: RunName
    tower: TowerSettings
    head: HeadSettings
    loss: LossSettings
    train: TrainSettings
    runtime: RuntimeSettings


def load_config(path: Path) -> JobConfig:
    """Read and validate a TOML recipe; unknown keys are errors."""

    with path.open("rb") as handle:
        return JobConfig.model_validate(tomllib.load(handle))


def with_device(config: JobConfig, device: str, precision: str) -> JobConfig:
    """The same recipe on another device / tower precision (for CPU checks)."""

    data = config.model_dump()
    data["runtime"]["device"] = device
    data["tower"]["precision"] = precision
    return JobConfig.model_validate(data)
