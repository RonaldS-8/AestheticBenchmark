"""Shared base class and field types for annotation, training and eval records.

Path and hash types are defined with the crawl schema and re-used here so
every stage validates them the same way.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from lora_sft.crawl.schema import RelativePath, Sha256, SiteKey

__all__ = [
    "IMAGE_ID_PATTERN",
    "ImageId",
    "MEDIA_ID_PATTERN",
    "MediaId",
    "PromptId",
    "RawScores",
    "Record",
    "RelativePath",
    "RunName",
    "SCHEMA_VERSION",
    "SCORE_MAX",
    "SCORE_MIN",
    "SchemaVersion",
    "Score",
    "Scores",
    "Sha256",
    "SiteKey",
    "VIDEO_ID_PATTERN",
    "VideoId",
]

SCHEMA_VERSION = "1.0"
SCORE_MIN = 1
SCORE_MAX = 5
IMAGE_ID_PATTERN = r"^photo_\d{6,}$"
VIDEO_ID_PATTERN = r"^video_\d{6,}$"
MEDIA_ID_PATTERN = r"^(photo|video)_\d{6,}$"

SchemaVersion = Literal["1.0"]
Score = Annotated[float, Field(ge=SCORE_MIN, le=SCORE_MAX)]
ImageId = Annotated[str, Field(pattern=IMAGE_ID_PATTERN)]
VideoId = Annotated[str, Field(pattern=VIDEO_ID_PATTERN)]
MediaId = Annotated[str, Field(pattern=MEDIA_ID_PATTERN)]
# Prompt template name plus version, e.g. ``rubric-score@v1`` for
# ``prompts/**/rubric-score.prompt.md`` at revision 1.
PromptId = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*@v\d+$")]
RunName = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]


class Record(BaseModel):
    """Base for pipeline records: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Scores(Record):
    """Aesthetic, technical and overall scores on the 1–5 scale."""

    aesthetic: Score
    technical: Score
    overall: Score


class RawScores(Record):
    """Aesthetic, technical and overall scores on any finite scale.

    Used where the scale is not ours: dataset MOS labels and outputs of
    external models such as DOVER. Correlation metrics are scale-free.
    """

    aesthetic: float = Field(allow_inf_nan=False)
    technical: float = Field(allow_inf_nan=False)
    overall: float = Field(allow_inf_nan=False)
