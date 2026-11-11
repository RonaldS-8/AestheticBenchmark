"""Parser for the Explainable Visual Aesthetics (EVA) rating table."""

from __future__ import annotations

import csv
import io
import statistics
from collections import defaultdict
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lora_sft.crawl.base import CaptureError, validate_capture
from lora_sft.crawl.dataset_schema import DatasetScoreRecord, ScoreSummary
from lora_sft.crawl.schema import CapturedResponse

EVA_HEADER = (
    "image_id",
    "user_id",
    "score",
    "difficulty",
    "visual",
    "composition",
    "quality",
    "semantic",
    "vote_time",
    "1",
    "2",
    "3",
    "4",
)
BinaryFlag = Annotated[int, Field(ge=0, le=1)]


class _EvaVote(BaseModel):
    """One validated row from EVA ``votes_filtered.csv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str = Field(pattern=r"^\d+$")
    user_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=10)
    difficulty: float = Field(ge=1, le=4)
    visual: float = Field(ge=1, le=4)
    composition: float = Field(ge=1, le=4)
    quality: float = Field(ge=1, le=4)
    semantic: float = Field(ge=1, le=4)
    vote_time: float = Field(ge=0)
    factor_1: BinaryFlag = Field(alias="1")
    factor_2: BinaryFlag = Field(alias="2")
    factor_3: BinaryFlag = Field(alias="3")
    factor_4: BinaryFlag = Field(alias="4")


def _summary(values: list[float], scale_min: float, scale_max: float) -> ScoreSummary:
    return ScoreSummary(
        mean=round(statistics.fmean(values), 6),
        standard_deviation=round(statistics.pstdev(values), 6),
        count=len(values),
        scale_min=scale_min,
        scale_max=scale_max,
    )


class EvaAdapter:
    """Turn one intact EVA score-table capture into per-image score records."""

    site = "eva"

    def parse_capture(
        self, capture: CapturedResponse, body: bytes
    ) -> list[DatasetScoreRecord]:
        """Validate and aggregate all source votes without changing scales."""

        validate_capture(capture, body, self.site)
        try:
            text = body.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise CaptureError("EVA 评分表不是有效 UTF-8") from exc

        reader = csv.DictReader(io.StringIO(text), delimiter="=")
        if tuple(reader.fieldnames or ()) != EVA_HEADER:
            raise CaptureError("EVA 评分表表头与已核对格式不一致")

        scores: dict[str, list[float]] = defaultdict(list)
        qualities: dict[str, list[float]] = defaultdict(list)
        seen_votes: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                vote = _EvaVote.model_validate(row)
            except ValidationError as exc:
                raise CaptureError(f"EVA 评分表第 {line_number} 行无效: {exc}") from exc
            key = (vote.image_id, vote.user_id)
            if key in seen_votes:
                raise CaptureError(
                    f"EVA 评分表第 {line_number} 行重复评分: "
                    f"{vote.image_id}/{vote.user_id}"
                )
            seen_votes.add(key)
            scores[vote.image_id].append(vote.score)
            qualities[vote.image_id].append(vote.quality)

        if not scores:
            raise CaptureError("EVA 评分表没有数据行")
        if (
            scores.keys() != qualities.keys()
        ):  # pragma: no cover - construction invariant
            raise AssertionError("EVA 分数聚合键不一致")

        return [
            DatasetScoreRecord(
                schema_version="1.0",
                capture_id=capture.capture_id,
                dataset=self.site,
                source_image_id=image_id,
                image_member=f"{image_id}.jpg",
                aesthetic=_summary(scores[image_id], 0, 10),
                technical=_summary(qualities[image_id], 1, 4),
                overall=None,
            )
            for image_id in sorted(scores, key=int)
        ]
