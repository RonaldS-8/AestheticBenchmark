"""Source-provided aesthetic and technical score records.

Research datasets often distribute one archive of images plus a table of
ratings rather than one directly downloadable URL per image. These records
preserve that score table without pretending it is a normal photo-site
engagement signal or a project ``AnnotationRecord``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from lora_sft.crawl.schema import RelativePath, Sha256, SiteKey

DATASET_SCORE_SCHEMA_VERSION = "1.0"


class _Record(BaseModel):
    """Immutable record that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreSummary(_Record):
    """Aggregate of source ratings on their original numeric scale."""

    mean: float
    standard_deviation: float = Field(ge=0)
    count: int = Field(gt=0)
    scale_min: float
    scale_max: float

    @model_validator(mode="after")
    def _check_scale(self) -> ScoreSummary:
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min 必须小于 scale_max")
        if not self.scale_min <= self.mean <= self.scale_max:
            raise ValueError(
                f"mean={self.mean} 超出评分区间 "
                f"[{self.scale_min}, {self.scale_max}]"
            )
        return self


class DatasetScoreRecord(_Record):
    """A source image with independent aesthetic and technical ratings."""

    schema_version: Literal["1.0"]
    capture_id: str = Field(min_length=1)
    dataset: SiteKey
    source_image_id: str = Field(min_length=1)
    image_member: RelativePath
    aesthetic: ScoreSummary
    technical: ScoreSummary
    overall: ScoreSummary | None


class DatasetImageRecord(_Record):
    """One extracted source image with a digest for leakage checks."""

    schema_version: Literal["1.0"]
    dataset: SiteKey
    source_image_id: str = Field(min_length=1)
    image_path: RelativePath
    sha256: Sha256
    format: Literal["jpeg"]
    size_bytes: int = Field(gt=0)
    has_eoi_marker: bool


class DatasetCrawlManifest(_Record):
    """Provenance and outputs of one score-table crawl."""

    schema_version: Literal["1.0"]
    dataset: SiteKey
    source_page_url: HttpUrl
    ratings_url: HttpUrl
    archive_urls: tuple[HttpUrl, ...] = Field(min_length=1)
    ratings_capture_id: str = Field(min_length=1)
    scores_path: RelativePath
    record_count: int = Field(gt=0)
    research_only: bool
    archive_image_count: int | None = Field(ge=1)
    excluded_unrated_count: int | None = Field(ge=0)
    excluded_unrated_path: RelativePath | None
    images_path: RelativePath | None
    image_record_count: int | None = Field(ge=1)
    jpeg_without_eoi_count: int | None = Field(ge=0)
