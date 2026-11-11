from __future__ import annotations

import pytest
import torch
from aes_head.config import LossSettings
from aes_head.losses import plcc_loss, rank_loss, score_loss

ONLY_MSE = LossSettings(mse_weight=1.0, plcc_weight=0.0, rank_weight=0.0)


def test_plcc_loss_is_zero_for_agreement_and_one_for_reversal() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert float(plcc_loss(2 * target + 1, target)) == pytest.approx(0.0, abs=1e-6)
    assert float(plcc_loss(-target, target)) == pytest.approx(1.0, abs=1e-6)


def test_rank_loss_counts_only_misordered_pairs() -> None:
    target = torch.tensor([1.0, 2.0, 3.0])

    assert float(rank_loss(torch.tensor([0.0, 1.0, 5.0]), target)) == 0.0
    # One misordered pair (gap 1) appears twice in the n*(n-1) ordered pairs.
    assert float(rank_loss(torch.tensor([1.0, 0.0, 5.0]), target)) == pytest.approx(
        2 / 6
    )


def test_score_loss_sums_columns_and_needs_a_batch() -> None:
    predicted = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    assert float(score_loss(predicted, target, ONLY_MSE)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="batch"):
        score_loss(predicted[:1], target[:1], ONLY_MSE)
    with pytest.raises(ValueError, match="形状"):
        score_loss(predicted, target[:, :1], ONLY_MSE)


def test_score_loss_gradients_flow() -> None:
    predicted = torch.randn(6, 2, requires_grad=True)
    target = torch.randn(6, 2)
    settings = LossSettings(mse_weight=1.0, plcc_weight=1.0, rank_weight=0.3)

    score_loss(predicted, target, settings).backward()

    assert predicted.grad is not None and bool(predicted.grad.abs().sum() > 0)
