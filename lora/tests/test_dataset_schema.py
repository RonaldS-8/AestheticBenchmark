from __future__ import annotations

import pytest
from pydantic import ValidationError

from lora_sft.crawl.dataset_schema import ScoreSummary


def test_score_summary_accepts_source_scale() -> None:
    summary = ScoreSummary(
        mean=3.5,
        standard_deviation=0.5,
        count=20,
        scale_min=1,
        scale_max=4,
    )

    assert summary.mean == 3.5


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mean": 5}, "超出评分区间"),
        ({"scale_min": 4}, "scale_min"),
        ({"count": 0}, "count"),
        ({"standard_deviation": -0.1}, "standard_deviation"),
    ],
)
def test_score_summary_rejects_invalid_values(
    changes: dict[str, float], message: str
) -> None:
    data = {
        "mean": 3,
        "standard_deviation": 0.5,
        "count": 20,
        "scale_min": 1,
        "scale_max": 4,
    }
    data.update(changes)

    with pytest.raises(ValidationError, match=message):
        ScoreSummary.model_validate(data)
