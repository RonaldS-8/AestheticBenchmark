"""Ridge-regression probe on frozen features.

For each score the ridge strength is chosen by k-fold cross-validation on the
train split (mean SRCC over folds), refit on all of train and scored once on
val. Features are standardized with train statistics; columns that are
constant on the training rows carry no information and are dropped. Val SRCC
gets a percentile bootstrap 95% interval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from aes_head.metrics import plcc, srcc
from aes_head.schema import DimensionMetrics

DEFAULT_ALPHAS = tuple(10.0**k for k in range(-1, 7))


class ProbeError(ValueError):
    """Raised for inconsistent probe inputs."""


@dataclass(frozen=True)
class Standardizer:
    """Train-set column statistics for z-scoring."""

    mean: torch.Tensor
    std: torch.Tensor
    keep: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> Standardizer:
        """Fit on training rows; zero-variance columns are marked to drop."""

        std = features.std(dim=0)
        keep = std > 1e-12
        if not bool(keep.any()):
            raise ProbeError("所有特征列在训练集上都是常数")
        return cls(features.mean(dim=0)[keep], std[keep], keep)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Z-score with the training statistics."""

        return (features[:, self.keep] - self.mean) / self.std


def ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    alphas: Sequence[float],
) -> torch.Tensor:
    """Ridge predictions ``[len(alphas), n_test]`` from one SVD of the train set."""

    if train_x.shape[0] != train_y.shape[0] or train_x.shape[1] != test_x.shape[1]:
        raise ProbeError("训练特征、标签和测试特征的形状不一致")
    scaler = Standardizer.fit(train_x)
    x = scaler.transform(train_x)
    z = scaler.transform(test_x)
    y_mean = train_y.mean()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    uty = u.T @ (train_y - y_mean)
    projected = z @ vh.T
    rows = [projected @ (s / (s**2 + alpha) * uty) + y_mean for alpha in alphas]
    return torch.stack(rows)


def fold_indices(n: int, folds: int, seed: int) -> list[torch.Tensor]:
    """Shuffle ``range(n)`` with ``seed`` and cut it into ``folds`` parts."""

    if not 2 <= folds <= n:
        raise ProbeError(f"折数 {folds} 必须在 2 到样本数 {n} 之间")
    generator = torch.Generator().manual_seed(seed)
    return list(torch.randperm(n, generator=generator).tensor_split(folds))


def select_alpha(
    x: torch.Tensor,
    y: torch.Tensor,
    alphas: Sequence[float],
    folds: int,
    seed: int,
) -> tuple[float, float]:
    """The alpha with the highest mean held-out SRCC, and that SRCC."""

    totals = [0.0] * len(alphas)
    parts = fold_indices(x.shape[0], folds, seed)
    for held in parts:
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        mask[held] = False
        predictions = ridge_predict(x[mask], y[mask], x[held], alphas)
        held_y = y[held].tolist()
        for k, row in enumerate(predictions):
            totals[k] += srcc(row.tolist(), held_y)
    means = [total / len(parts) for total in totals]
    best = max(range(len(alphas)), key=lambda k: means[k])
    return alphas[best], means[best]


def bootstrap_srcc_ci(
    predicted: Sequence[float],
    labels: Sequence[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of SRCC."""

    rng = random.Random(seed)
    n = len(labels)
    values = []
    for _ in range(samples):
        picks = [rng.randrange(n) for _ in range(n)]
        values.append(srcc([predicted[i] for i in picks], [labels[i] for i in picks]))
    values.sort()
    return values[int(0.025 * (samples - 1))], values[int(0.975 * (samples - 1))]


@dataclass(frozen=True)
class DimensionProbe:
    """Outcome of probing one score."""

    alpha: float
    cv_srcc: float
    val: DimensionMetrics
    val_srcc_ci95: tuple[float, float]


def probe_dimension(
    train_x: torch.Tensor,
    train_target: Sequence[float],
    val_x: torch.Tensor,
    val_means: Sequence[float],
    val_reported: Sequence[float],
    *,
    alphas: Sequence[float],
    folds: int,
    bootstrap: int,
    seed: int,
) -> DimensionProbe:
    """Pick alpha on train, refit, and score val against the vote means.

    ``train_target`` is the unrounded 1-5 target; ``val_means`` are raw EVA
    vote means (for SRCC/PLCC) and ``val_reported`` the rounded 1-5 targets
    (for MAE), exactly as the LoRA eval reports them.
    """

    if not len(val_means) == len(val_reported) == val_x.shape[0]:
        raise ProbeError("val 特征与标签数量不一致")
    y = torch.tensor(train_target, dtype=torch.float64)
    alpha, cv_srcc = select_alpha(train_x, y, alphas, folds, seed)
    predicted = ridge_predict(train_x, y, val_x, [alpha])[0].tolist()
    metrics = DimensionMetrics(
        srcc=srcc(predicted, val_means),
        plcc=plcc(predicted, val_means),
        mae_1_5=sum(abs(p - t) for p, t in zip(predicted, val_reported, strict=True))
        / len(predicted),
    )
    interval = bootstrap_srcc_ci(predicted, val_means, bootstrap, seed)
    return DimensionProbe(alpha, cv_srcc, metrics, interval)
