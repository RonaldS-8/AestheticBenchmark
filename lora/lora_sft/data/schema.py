"""Annotation records and dataset splits.

One ``AnnotationRecord`` per image holds every annotator's scores; SFT samples
are generated from it. ``SplitManifest`` fixes which images are used for
training, validation and the frozen evaluation set.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import AwareDatetime, Field, model_validator

from lora_sft.common import (
    SCORE_MAX,
    SCORE_MIN,
    ImageId,
    MediaId,
    Record,
    RelativePath,
    SchemaVersion,
    Scores,
    Sha256,
    SiteKey,
)
from lora_sft.crawl.schema import LicenseType

MIN_EVAL_ANNOTATORS = 3


class SplitError(ValueError):
    """Raised when a split leaks images or misses annotations."""


class Category(StrEnum):
    """Photo categories the product sees; balanced across splits."""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    FAMILY_DAILY = "family_daily"
    NIGHT = "night"
    PET = "pet"
    FOOD = "food"
    ARCHITECTURE = "architecture"
    STREET = "street"
    OTHER = "other"


class Dimension(StrEnum):
    """Judgement dimensions; the first four feed the aesthetic score."""

    COMPOSITION = "composition"
    LIGHTING = "lighting"
    COLOR = "color"
    SUBJECT = "subject"
    FOCUS = "focus"
    EXPOSURE = "exposure"
    NOISE = "noise"
    BLUR = "blur"


class Annotation(Record):
    """One annotator's judgement of one image."""

    annotator: str = Field(pattern=r"^A\d{2,}$")
    aesthetic: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    technical: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    overall: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    reason: str = Field(min_length=1, max_length=100)
    highlights: tuple[Dimension, ...]
    issues: tuple[Dimension, ...]
    annotated_at: AwareDatetime

    @model_validator(mode="after")
    def _check_dimensions(self) -> Annotation:
        for name, dims in (("highlights", self.highlights), ("issues", self.issues)):
            if len(set(dims)) != len(dims):
                raise ValueError(f"{name} 中有重复维度")
        both = set(self.highlights) & set(self.issues)
        if both:
            raise ValueError(f"维度不能同时是亮点和问题: {sorted(both)}")
        return self


class AnnotationSource(Record):
    """Link back to the crawled photo."""

    site: SiteKey
    site_photo_id: str = Field(min_length=1)
    license_type: LicenseType


def mean_scores(annotations: Iterable[Annotation]) -> Scores:
    """Average annotator scores, rounded to two decimals."""

    items = list(annotations)
    if not items:
        raise ValueError("没有标注，无法计算均分")
    n = len(items)
    return Scores(
        aesthetic=round(sum(a.aesthetic for a in items) / n, 2),
        technical=round(sum(a.technical for a in items) / n, 2),
        overall=round(sum(a.overall for a in items) / n, 2),
    )


class AnnotationRecord(Record):
    """All annotations of one image plus their stored mean."""

    schema_version: SchemaVersion
    image_id: ImageId
    image_path: RelativePath
    image_sha256: Sha256
    category: Category
    source: AnnotationSource
    guideline_version: str = Field(pattern=r"^v\d+$")
    annotations: tuple[Annotation, ...] = Field(min_length=1)
    mean: Scores

    @model_validator(mode="after")
    def _check_consistency(self) -> AnnotationRecord:
        if PurePosixPath(self.image_path).stem != self.image_id:
            raise ValueError(
                f"图片文件名 {self.image_path!r} 必须与 image_id "
                f"{self.image_id!r} 一致"
            )
        annotators = [a.annotator for a in self.annotations]
        if len(set(annotators)) != len(annotators):
            raise ValueError(f"{self.image_id}: 同一标注者重复标注")
        expected = mean_scores(self.annotations)
        if self.mean != expected:
            raise ValueError(
                f"{self.image_id}: mean 与标注不一致，应为 " f"{expected.model_dump()}"
            )
        return self


def eval_fingerprint(image_ids: Iterable[str]) -> str:
    """Order-independent SHA-256 of the eval image ids."""

    joined = "\n".join(sorted(image_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class SplitManifest(Record):
    """Photo or video ids per split; the eval split never changes within a
    version. One manifest holds one media type."""

    schema_version: SchemaVersion
    split_version: str = Field(pattern=r"^split-v\d+$")
    created_at: AwareDatetime
    seed: int = Field(ge=0)
    train: tuple[MediaId, ...] = Field(min_length=1)
    val: tuple[MediaId, ...] = Field(min_length=1)
    eval: tuple[MediaId, ...] = Field(min_length=1)
    eval_fingerprint: Sha256

    @model_validator(mode="after")
    def _check_disjoint(self) -> SplitManifest:
        seen: dict[str, str] = {}
        for split, ids in (
            ("train", self.train),
            ("val", self.val),
            ("eval", self.eval),
        ):
            for image_id in ids:
                if image_id in seen:
                    raise ValueError(
                        f"{image_id} 同时出现在 {seen[image_id]} 和 {split}"
                    )
                seen[image_id] = split
        prefixes = {image_id.partition("_")[0] for image_id in seen}
        if len(prefixes) != 1:
            raise ValueError("一个划分不能混用照片和视频")
        if eval_fingerprint(self.eval) != self.eval_fingerprint:
            raise ValueError("eval_fingerprint 与 eval 列表不符，评测集被改动")
        return self


def check_split(
    manifest: SplitManifest, records: Mapping[str, AnnotationRecord]
) -> None:
    """Cross-check a split against annotation records before training.

    Every id needs a record, eval images need ``MIN_EVAL_ANNOTATORS``
    annotations, and no image file (by SHA-256) may appear in both eval and
    train/val under different ids.
    """

    for image_id in (*manifest.train, *manifest.val, *manifest.eval):
        if image_id not in records:
            raise SplitError(f"{image_id} 没有标注记录")
    for image_id in manifest.eval:
        count = len(records[image_id].annotations)
        if count < MIN_EVAL_ANNOTATORS:
            raise SplitError(
                f"评测图片 {image_id} 只有 {count} 人标注，"
                f"至少需要 {MIN_EVAL_ANNOTATORS} 人"
            )
    eval_hashes = {records[i].image_sha256: i for i in manifest.eval}
    for image_id in (*manifest.train, *manifest.val):
        twin = eval_hashes.get(records[image_id].image_sha256)
        if twin is not None:
            raise SplitError(f"{image_id} 与评测图片 {twin} 是同一文件")


def ensure_not_eval(image_ids: Iterable[str], manifest: SplitManifest) -> None:
    """Reject training samples that use an eval image.

    Covers samples outside the split too (e.g. site-rating labels for
    unannotated photos). Relies on the crawl stage giving each SHA-256 one
    ``image_id``.
    """

    eval_ids = set(manifest.eval)
    leaked = sorted(set(image_ids) & eval_ids)
    if leaked:
        raise SplitError(f"训练样本使用了评测图片: {leaked[:5]}")
