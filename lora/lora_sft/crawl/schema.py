"""Crawl-stage record schemas for the photo-aesthetics dataset.

Records flow through three types, each stored as one JSON object per line:

1. ``CapturedResponse``: one raw HTTP response saved during packet capture or
   crawling; the body itself lives on disk under ``data/raw``.
2. ``PhotoCandidate``: one photo parsed out of a capture by a site adapter. It
   carries metadata and a license but no downloaded file yet.
3. ``PhotoRecord``: a candidate whose image has been downloaded and hashed.
   ``image_id``, ``image.path`` and ``source`` feed the annotation record.

Only photos whose license allows model training may enter the research
pipeline. Commercial runs additionally require commercial permission; pass an
explicit ``TrainingPurpose`` to ``ensure_trainable`` before downloading.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

SCHEMA_VERSION = "1.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
SITE_PATTERN = r"^[a-z0-9_]+$"
IMAGE_SUFFIXES: dict[str, frozenset[str]] = {
    "jpeg": frozenset({".jpg", ".jpeg"}),
    "png": frozenset({".png"}),
    "webp": frozenset({".webp"}),
}


class ComplianceError(Exception):
    """Raised when a photo is not licensed for training or commercial use."""


class TrainingPurpose(StrEnum):
    """Declared use checked before a photo is downloaded."""

    RESEARCH = "research"
    COMMERCIAL = "commercial"


def _check_relative_posix(value: str) -> str:
    """Accept only normalized relative POSIX paths inside the data root."""

    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ":" in value
        or ".." in path.parts
        or value != path.as_posix()
        or value == "."
    ):
        raise ValueError(f"路径必须是数据根目录下规范的相对 POSIX 路径: {value!r}")
    return value


RelativePath = Annotated[str, AfterValidator(_check_relative_posix)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
SiteKey = Annotated[str, Field(pattern=SITE_PATTERN)]


class _Record(BaseModel):
    """Base for crawl records: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LicenseType(StrEnum):
    """License families seen on photo sites; ``UNKNOWN`` is never usable."""

    CC0 = "cc0"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC_BY_ND = "cc_by_nd"
    CC_BY_NC = "cc_by_nc"
    CC_BY_NC_SA = "cc_by_nc_sa"
    CC_BY_NC_ND = "cc_by_nc_nd"
    SITE_LICENSE = "site_license"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


NON_COMMERCIAL_LICENSES = frozenset(
    {LicenseType.CC_BY_NC, LicenseType.CC_BY_NC_SA, LicenseType.CC_BY_NC_ND}
)


class LicenseInfo(_Record):
    """License of one photo, checked by a person against the site's terms."""

    license_type: LicenseType
    training_allowed: bool
    commercial_allowed: bool
    terms_url: HttpUrl
    verified_by: str = Field(min_length=1)
    verified_at: date

    @model_validator(mode="after")
    def _check_consistency(self) -> LicenseInfo:
        if self.license_type is LicenseType.UNKNOWN and (
            self.training_allowed or self.commercial_allowed
        ):
            raise ValueError("授权类型未知时不能标记为允许训练或商用")
        if self.license_type in NON_COMMERCIAL_LICENSES and self.commercial_allowed:
            raise ValueError(f"{self.license_type.value} 不允许商用")
        return self


class SiteRating(_Record):
    """Score the site itself shows (e.g. a community vote average)."""

    mean: float
    count: int = Field(gt=0)
    scale_min: float
    scale_max: float

    @model_validator(mode="after")
    def _check_range(self) -> SiteRating:
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min 必须小于 scale_max")
        if not self.scale_min <= self.mean <= self.scale_max:
            raise ValueError(
                f"mean={self.mean} 超出评分区间 "
                f"[{self.scale_min}, {self.scale_max}]"
            )
        return self


class SiteSignals(_Record):
    """Engagement shown on the site; weak labels, never ground truth.

    Every field must be given explicitly: ``None`` means the site does not
    expose that value, which is different from zero.
    """

    views: int | None = Field(ge=0)
    likes: int | None = Field(ge=0)
    favorites: int | None = Field(ge=0)
    comments: int | None = Field(ge=0)
    rating: SiteRating | None


class SourceInfo(_Record):
    """Where a photo came from and under which license."""

    site: SiteKey
    site_photo_id: str = Field(min_length=1)
    page_url: HttpUrl
    image_url: HttpUrl
    author: str | None
    license: LicenseInfo


class CapturedResponse(_Record):
    """One raw HTTP response saved during packet capture or crawling."""

    schema_version: Literal["1.0"]
    capture_id: str = Field(min_length=1)
    site: SiteKey
    method: Literal["GET", "POST"]
    url: HttpUrl
    status_code: int = Field(ge=100, le=599)
    content_type: str = Field(min_length=1)
    captured_at: AwareDatetime
    body_path: RelativePath
    body_sha256: Sha256
    body_size: int = Field(ge=0)


class ImageFile(_Record):
    """A downloaded image file, relative to the data root."""

    path: RelativePath
    sha256: Sha256
    format: Literal["jpeg", "png", "webp"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_suffix(self) -> ImageFile:
        suffix = PurePosixPath(self.path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES[self.format]:
            raise ValueError(f"文件后缀 {suffix!r} 与格式 {self.format} 不一致")
        return self


class PhotoCandidate(_Record):
    """A photo parsed from a capture, before its image is downloaded."""

    schema_version: Literal["1.0"]
    capture_id: str = Field(min_length=1)
    source: SourceInfo
    signals: SiteSignals
    site_tags: tuple[str, ...]


class PhotoRecord(PhotoCandidate):
    """A candidate whose image has been downloaded and hashed."""

    image_id: str = Field(pattern=r"^photo_\d{6,}$")
    image: ImageFile
    downloaded_at: AwareDatetime

    @model_validator(mode="after")
    def _check_image_name(self) -> PhotoRecord:
        if PurePosixPath(self.image.path).stem != self.image_id:
            raise ValueError(
                f"图片文件名 {self.image.path!r} 必须与 image_id "
                f"{self.image_id!r} 一致"
            )
        return self


def ensure_trainable(photo: PhotoCandidate, purpose: TrainingPurpose) -> None:
    """Raise unless the photo may be used for the declared training purpose.

    Applies to both training and evaluation data. Call it on candidates before
    downloading so unusable images are never fetched. The purpose is mandatory
    so a research-only source cannot silently enter a commercial run.
    """

    license_info = photo.source.license
    ref = f"{photo.source.site}/{photo.source.site_photo_id}"
    if not license_info.training_allowed:
        raise ComplianceError(
            f"{ref}: 授权不允许用于模型训练 ({license_info.terms_url})"
        )
    if purpose is TrainingPurpose.COMMERCIAL and not license_info.commercial_allowed:
        raise ComplianceError(f"{ref}: 授权不允许商用 ({license_info.terms_url})")
