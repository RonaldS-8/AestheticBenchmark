from __future__ import annotations

import pytest
import torch
from aes_head.metrics import srcc
from aes_head.probe import (
    ProbeError,
    Standardizer,
    bootstrap_srcc_ci,
    fold_indices,
    probe_dimension,
    ridge_predict,
    select_alpha,
)


def _linear_data(n: int, d: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator, dtype=torch.float64)
    weights = torch.randn(d, generator=generator, dtype=torch.float64)
    noise = 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    return x, x @ weights + noise


def test_ridge_recovers_a_linear_signal() -> None:
    x, y = _linear_data(240, 20)

    predictions = ridge_predict(x[:200], y[:200], x[200:], [0.1, 1e6])

    assert predictions.shape == (2, 40)
    assert srcc(predictions[0].tolist(), y[200:].tolist()) > 0.95
    # A huge penalty shrinks everything towards the training mean.
    assert float(predictions[1].std()) < 0.1 * float(predictions[0].std())


def test_ridge_rejects_mismatched_shapes() -> None:
    x, y = _linear_data(20, 4)

    with pytest.raises(ProbeError, match="形状"):
        ridge_predict(x, y[:10], x, [1.0])
    with pytest.raises(ProbeError, match="形状"):
        ridge_predict(x, y, x[:, :3], [1.0])


def test_standardizer_drops_constant_columns() -> None:
    x = torch.tensor([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0]], dtype=torch.float64)

    scaler = Standardizer.fit(x)

    assert scaler.keep.tolist() == [True, False, True]
    assert scaler.transform(x).shape == (2, 2)
    with pytest.raises(ProbeError, match="常数"):
        Standardizer.fit(torch.ones(3, 2, dtype=torch.float64))


def test_fold_indices_partition_all_rows() -> None:
    parts = fold_indices(11, 3, seed=1)

    assert sorted(torch.cat(parts).tolist()) == list(range(11))
    assert len(parts) == 3
    with pytest.raises(ProbeError, match="折数"):
        fold_indices(3, 5, seed=1)


def test_select_alpha_prefers_light_penalty_on_clean_signal() -> None:
    x, y = _linear_data(150, 10)

    alpha, cv = select_alpha(x, y, [0.01, 1e5], folds=5, seed=0)

    assert alpha == 0.01
    assert cv > 0.95


def test_bootstrap_interval_brackets_the_estimate() -> None:
    labels = [float(i) for i in range(50)]
    predicted = [v + (3 if i % 4 == 0 else -2) for i, v in enumerate(labels)]

    low, high = bootstrap_srcc_ci(predicted, labels, samples=200, seed=0)

    assert low <= srcc(predicted, labels) <= high <= 1


def test_probe_dimension_scores_val_against_means() -> None:
    x, y = _linear_data(120, 8)
    target = (3 + y / y.abs().max()).tolist()

    outcome = probe_dimension(
        x[:100],
        target[:100],
        x[100:],
        target[100:],
        [round(v, 1) for v in target[100:]],
        alphas=[0.01, 10.0],
        folds=4,
        bootstrap=100,
        seed=0,
    )

    assert outcome.val.srcc > 0.9
    assert outcome.val.mae_1_5 < 0.1
    assert outcome.val_srcc_ci95[0] <= outcome.val.srcc
    with pytest.raises(ProbeError, match="数量"):
        probe_dimension(
            x[:100],
            target[:100],
            x[100:],
            target[100:],
            target[100:110],
            alphas=[1.0],
            folds=4,
            bootstrap=100,
            seed=0,
        )
