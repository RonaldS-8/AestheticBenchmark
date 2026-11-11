"""Score metrics against human labels (copy of the main repository's).

SRCC is Spearman's rank correlation with average ranks for ties; PLCC is the
plain Pearson correlation (no logistic fitting, since only ranks and linear
agreement matter for the comparison). ``tests/test_sync_with_main.py`` checks
these give the same numbers as ``lora_sft/eval/metrics.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class MetricsError(ValueError):
    """Raised when a metric is undefined for the given data."""


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2 + 1
        for k in range(start, end + 1):
            ranks[order[k]] = average
        start = end + 1
    return ranks


def plcc(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; undefined for constant inputs."""

    if len(x) != len(y) or len(x) < 2:
        raise MetricsError("相关系数需要至少 2 对等长数据")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise MetricsError("有一组分数全部相同，相关系数无定义")
    # Rounding can push a perfect correlation to 1.0000000000000002.
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def srcc(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""

    return plcc(_ranks(x), _ranks(y))


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, ``q`` in (0, 100]."""

    if not values:
        raise MetricsError("没有数据")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]
