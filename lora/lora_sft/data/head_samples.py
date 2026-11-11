"""Samples for the DOVER-style aesthetic head.

The head regresses scores directly from pixels, so a sample is just an image
and its targets. Targets come from human annotation means or, for weak-label
pre-training, from the score a photo site shows (overall only).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import Field, model_validator

from lora_sft.common import (
    SCORE_MAX,
    SCORE_MIN,
    ImageId,
    Record,
    RelativePath,
    SchemaVersion,
    Score,
)
from lora_sft.crawl.schema import SiteRating
from lora_sft.data.samples import SampleMismatchError
from lora_sft.data.schema import AnnotationRecord


class LabelSource(StrEnum):
    """Where head targets come from."""

    HUMAN_MEAN = "human_mean"
    SITE_RATING = "site_rating"


class HeadTargets(Record):
    """Regression targets; ``None`` means no label for that dimension."""

    aesthetic: Score | None
    technical: Score | None
    overall: Score


class HeadSample(Record):
    """One image with its regression targets and loss weight."""

    schema_version: SchemaVersion
    image_id: ImageId
    image_path: RelativePath
    label_source: LabelSource
    targets: HeadTargets
    weight: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _check_targets(self) -> HeadSample:
        if PurePosixPath(self.image_path).stem != self.image_id:
            raise ValueError(
                f"图片 {self.image_path!r} 与 image_id {self.image_id!r} 不一致"
            )
        partial = self.targets.aesthetic is None or self.targets.technical is None
        if self.label_source is LabelSource.HUMAN_MEAN and partial:
            raise ValueError("人工均分样本必须有美学、技术、总体三项目标")
        if self.label_source is LabelSource.SITE_RATING and (
            self.targets.aesthetic is not None or self.targets.technical is not None
        ):
            raise ValueError("站点评分只有总体分，aesthetic/technical 必须为 None")
        return self


def site_rating_to_score(rating: SiteRating) -> float:
    """Map a site's rating linearly onto the 1–5 scale, two decimals."""

    span = rating.scale_max - rating.scale_min
    share = (rating.mean - rating.scale_min) / span
    return round(SCORE_MIN + share * (SCORE_MAX - SCORE_MIN), 2)


def check_head_sample_matches_annotation(
    sample: HeadSample, record: AnnotationRecord
) -> None:
    """Ensure a human-label head sample carries the annotation means."""

    if sample.label_source is not LabelSource.HUMAN_MEAN:
        raise SampleMismatchError(f"{sample.image_id} 不是人工均分样本")
    if sample.image_id != record.image_id:
        raise SampleMismatchError(
            f"样本对应 {sample.image_id}，标注记录是 {record.image_id}"
        )
    expected = record.mean.model_dump()
    actual = sample.targets.model_dump()
    if actual != expected:
        raise SampleMismatchError(
            f"{sample.image_id} 的目标应为 {expected}，实际 {actual}"
        )
