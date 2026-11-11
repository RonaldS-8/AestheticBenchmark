"""Two score heads on standardized tower features, and their checkpoints.

Features are z-scored with train statistics stored in the model (a column
that is constant on train has nothing to centre on and maps to 0). Each
score has its own MLP predicting a z-score; the stored train target mean and
std turn it back into the 1-5 scale.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from qv_head.config import JobConfig
from qv_head.data import sha256_file
from qv_head.schema import EVA_DIMENSIONS

CHECKPOINT_FORMAT = 1


class HeadError(ValueError):
    """Raised for a missing, changed or incompatible checkpoint."""


class FeatureHead(nn.Module):
    """Standardize features, then one MLP per score."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        target_mean: Sequence[float],
        target_std: Sequence[float],
    ) -> None:
        super().__init__()
        if feature_mean.shape != (feature_dim,) or feature_std.shape != (feature_dim,):
            raise HeadError("特征均值、标准差的形状与特征维度不一致")
        std = torch.where(
            feature_std > 1e-12, feature_std, torch.ones_like(feature_std)
        )
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", std.float().clone())
        self.register_buffer("target_mean", torch.tensor(list(target_mean)))
        self.register_buffer("target_std", torch.tensor(list(target_std)))
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in EVA_DIMENSIONS
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` features -> ``[B, 2]`` z-scores (aesthetic, technical)."""

        x = (features - self.feature_mean) / self.feature_std
        return torch.cat([head(x) for head in self.heads], dim=1)

    def to_scores(self, z: torch.Tensor) -> torch.Tensor:
        """z-scores -> 1-5 scores (not clipped)."""

        return z * self.target_std + self.target_mean


def build_head(
    config: JobConfig, train_features: torch.Tensor, train_targets: torch.Tensor
) -> FeatureHead:
    """A fresh head with feature and target statistics of the train rows."""

    return FeatureHead(
        train_features.shape[1],
        hidden_dim=config.head.hidden_dim,
        dropout=config.head.dropout,
        feature_mean=train_features.mean(dim=0),
        feature_std=train_features.std(dim=0),
        target_mean=train_targets.mean(dim=0).tolist(),
        target_std=train_targets.std(dim=0).tolist(),
    )


def save_head(
    path: Path, head: FeatureHead, config: JobConfig, tower_sha256: str
) -> None:
    """Write the head and what it was trained on; replaces atomically."""

    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": config.model_dump(mode="json"),
            "feature_dim": int(head.feature_mean.shape[0]),
            "tower_sha256": tower_sha256,
            "state_dict": head.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_head(path: Path, sha256: str) -> tuple[FeatureHead, JobConfig, str]:
    """Rebuild a head after checking the file hash; returns its tower hash too."""

    if not path.is_file():
        raise HeadError(f"检查点不存在: {path}")
    if sha256_file(path) != sha256:
        raise HeadError(f"检查点哈希与训练记录不符: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != CHECKPOINT_FORMAT:
        raise HeadError(f"检查点格式 {saved['format']} 不支持")
    config = JobConfig.model_validate(saved["config"])
    dim = saved["feature_dim"]
    head = FeatureHead(
        dim,
        hidden_dim=config.head.hidden_dim,
        dropout=config.head.dropout,
        feature_mean=torch.zeros(dim),
        feature_std=torch.ones(dim),
        target_mean=[0.0] * len(EVA_DIMENSIONS),
        target_std=[1.0] * len(EVA_DIMENSIONS),
    )
    head.load_state_dict(saved["state_dict"], strict=True)
    return head, config, saved["tower_sha256"]
