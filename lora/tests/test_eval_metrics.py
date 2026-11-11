from __future__ import annotations

import pytest

from lora_sft.eval.metrics import (
    MetricsError,
    pairwise_agreement,
    percentile,
    plcc,
    srcc,
)


def test_plcc_perfect_and_inverse() -> None:
    assert plcc([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert plcc([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_plcc_stays_within_bounds_despite_rounding() -> None:
    values = [5.6, 6.3, 7.1, 8.2, 9.0, 5.2]

    assert plcc(values, [1 + 4 * v / 10 for v in values]) <= 1.0
    assert plcc(values, [-v for v in values]) >= -1.0


def test_srcc_is_rank_based_with_ties() -> None:
    assert srcc([1, 10, 100], [1, 2, 3]) == pytest.approx(1.0)
    # ranks of [1, 2, 2, 3] are [1, 2.5, 2.5, 4]
    assert srcc([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, rel=1e-6)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([1], [1], "至少 2 对"),
        ([1, 2], [1], "至少 2 对"),
        ([2, 2, 2], [1, 2, 3], "全部相同"),
    ],
)
def test_correlation_undefined(x: list[float], y: list[float], message: str) -> None:
    with pytest.raises(MetricsError, match=message):
        plcc(x, y)


def test_pairwise_agreement() -> None:
    labels = [1.0, 2.0, 3.0, 3.1]
    predicted = [0.1, 0.3, 0.2, 0.2]

    # pairs with gap >= 1: (0,1) hit, (0,2) hit, (0,3) hit, (1,2) miss, (1,3) miss
    assert pairwise_agreement(predicted, labels, 1.0) == pytest.approx(3 / 5)


def test_pairwise_agreement_counts_ties_as_misses() -> None:
    assert pairwise_agreement([2.0, 2.0], [1.0, 3.0], 0.5) == 0.0


def test_pairwise_agreement_needs_pairs() -> None:
    with pytest.raises(MetricsError, match="没有人工分差"):
        pairwise_agreement([1, 2], [1.0, 1.1], 0.5)


def test_percentile() -> None:
    values = [5, 1, 4, 2, 3]

    assert percentile(values, 50) == 3
    assert percentile(values, 95) == 5
    assert percentile(values, 1) == 1
    with pytest.raises(MetricsError, match="没有数据"):
        percentile([], 50)
