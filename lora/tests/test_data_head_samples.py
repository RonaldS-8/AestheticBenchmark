from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from lora_sft.crawl.schema import SiteRating
from lora_sft.data.head_samples import (
    HeadSample,
    LabelSource,
    check_head_sample_matches_annotation,
    site_rating_to_score,
)
from lora_sft.data.samples import SampleMismatchError
from lora_sft.data.schema import AnnotationRecord


def human_sample(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "image_id": "photo_000123",
        "image_path": "images/photo_000123.jpg",
        "label_source": "human_mean",
        "targets": {"aesthetic": 4.33, "technical": 2.67, "overall": 3.67},
        "weight": 1.0,
    }
    data.update(overrides)
    return data


def site_sample(**overrides: object) -> dict[str, Any]:
    targets = {"aesthetic": None, "technical": None, "overall": 3.96}
    return human_sample(label_source="site_rating", targets=targets, weight=0.5) | (
        overrides
    )


def test_human_sample_is_valid() -> None:
    sample = HeadSample.model_validate(human_sample())

    assert sample.label_source is LabelSource.HUMAN_MEAN
    assert sample.targets.technical == 2.67


def test_site_sample_is_valid() -> None:
    sample = HeadSample.model_validate(site_sample())

    assert sample.targets.aesthetic is None


@pytest.mark.parametrize("missing", ["aesthetic", "technical"])
def test_human_sample_needs_all_targets(missing: str) -> None:
    data = human_sample()
    data["targets"][missing] = None

    with pytest.raises(ValidationError, match="三项目标"):
        HeadSample.model_validate(data)


@pytest.mark.parametrize("extra", ["aesthetic", "technical"])
def test_site_sample_is_overall_only(extra: str) -> None:
    data = site_sample()
    data["targets"][extra] = 3.0

    with pytest.raises(ValidationError, match="只有总体分"):
        HeadSample.model_validate(data)


@pytest.mark.parametrize(
    ("weight", "valid"), [(1.0, True), (0.01, True), (0.0, False), (1.01, False)]
)
def test_weight_bounds(weight: float, valid: bool) -> None:
    data = human_sample(weight=weight)

    if valid:
        assert HeadSample.model_validate(data).weight == weight
    else:
        with pytest.raises(ValidationError, match="weight"):
            HeadSample.model_validate(data)


def test_target_out_of_scale_rejected() -> None:
    data = human_sample()
    data["targets"]["overall"] = 5.5

    with pytest.raises(ValidationError, match="overall"):
        HeadSample.model_validate(data)


def test_image_path_must_match_id() -> None:
    with pytest.raises(ValidationError, match="与 image_id"):
        HeadSample.model_validate(human_sample(image_path="images/photo_000999.jpg"))


@pytest.mark.parametrize(
    ("mean", "expected"), [(1.0, 1.0), (10.0, 5.0), (5.5, 3.0), (7.4, 3.84)]
)
def test_site_rating_to_score(mean: float, expected: float) -> None:
    rating = SiteRating(mean=mean, count=10, scale_min=1, scale_max=10)

    assert site_rating_to_score(rating) == expected


def test_head_sample_matches_annotation(annotation_data: dict[str, Any]) -> None:
    check_head_sample_matches_annotation(
        HeadSample.model_validate(human_sample()),
        AnnotationRecord.model_validate(annotation_data),
    )


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (site_sample(), "不是人工均分样本"),
        (
            human_sample(image_id="photo_000124", image_path="images/photo_000124.jpg"),
            "标注记录是 photo_000123",
        ),
        (
            human_sample(
                targets={"aesthetic": 4.3, "technical": 2.67, "overall": 3.67}
            ),
            "目标应为",
        ),
    ],
)
def test_head_sample_mismatch(
    annotation_data: dict[str, Any], sample: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleMismatchError, match=message):
        check_head_sample_matches_annotation(
            HeadSample.model_validate(sample),
            AnnotationRecord.model_validate(annotation_data),
        )
