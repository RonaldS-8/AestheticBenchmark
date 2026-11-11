"""Batch losses for score regression on z-scored targets.

PLCC and rank terms only see order and linear agreement inside the batch;
the MSE term pins the scale so predictions stay on the target scale.
"""

from __future__ import annotations

import torch

from qv_head.config import LossSettings


def plcc_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(1 - PLCC) / 2`` over a batch: 0 for perfect agreement, 1 for reversed."""

    p = predicted - predicted.mean()
    t = target - target.mean()
    correlation = (p * t).sum() / (p.norm() * t.norm() + 1e-8)
    return (1 - correlation) / 2


def rank_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean hinge over pairs the prediction orders against the target."""

    n = predicted.shape[0]
    gaps = predicted[:, None] - predicted[None, :]
    order = torch.sign(target[:, None] - target[None, :])
    violations = torch.relu(-gaps * order)
    return violations.sum() / (n * (n - 1))


def score_loss(
    predicted: torch.Tensor, target: torch.Tensor, settings: LossSettings
) -> torch.Tensor:
    """Weighted sum over score columns of MSE, PLCC and rank terms.

    ``predicted`` and ``target`` are ``[batch, scores]`` in z units.
    """

    if predicted.shape != target.shape or predicted.shape[0] < 2:
        raise ValueError("预测与目标形状必须相同，且 batch 至少 2 张")
    total = predicted.new_zeros(())
    for column in range(predicted.shape[1]):
        p, t = predicted[:, column], target[:, column]
        total = total + settings.mse_weight * torch.mean((p - t) ** 2)
        total = total + settings.plcc_weight * plcc_loss(p, t)
        total = total + settings.rank_weight * rank_loss(p, t)
    return total
