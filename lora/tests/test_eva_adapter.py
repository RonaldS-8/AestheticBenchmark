from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from lora_sft.crawl.adapters.eva import EvaAdapter
from lora_sft.crawl.base import CaptureError
from lora_sft.crawl.schema import CapturedResponse

FIXTURE = Path(__file__).parent / "fixtures" / "eva_votes_filtered.csv"


def capture_for(body: bytes, capture_data: dict[str, Any]) -> CapturedResponse:
    capture_data.update(
        capture_id="eva-test-ratings",
        site="eva",
        url="https://raw.githubusercontent.com/example/votes_filtered.csv",
        content_type="text/csv",
        body_path="raw/eva/votes_filtered.csv",
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_size=len(body),
    )
    return CapturedResponse.model_validate(capture_data)


def test_eva_adapter_aggregates_dual_scores(capture_data: dict[str, Any]) -> None:
    body = FIXTURE.read_bytes()

    records = EvaAdapter().parse_capture(capture_for(body, capture_data), body)

    assert [record.source_image_id for record in records] == ["71", "106"]
    first = records[0]
    assert first.aesthetic.mean == 6.0
    assert first.aesthetic.standard_deviation == 2.0
    assert first.aesthetic.count == 2
    assert (first.aesthetic.scale_min, first.aesthetic.scale_max) == (0, 10)
    assert first.technical.mean == 3.0
    assert first.technical.standard_deviation == 1.0
    assert (first.technical.scale_min, first.technical.scale_max) == (1, 4)
    assert first.overall is None
    assert first.image_member == "71.jpg"


def test_eva_adapter_rejects_wrong_header(capture_data: dict[str, Any]) -> None:
    body = FIXTURE.read_bytes().replace(b"image_id", b"photo_id", 1)

    with pytest.raises(CaptureError, match="表头"):
        EvaAdapter().parse_capture(capture_for(body, capture_data), body)


def test_eva_adapter_rejects_empty_table(capture_data: dict[str, Any]) -> None:
    body = FIXTURE.read_bytes().splitlines(keepends=True)[0]

    with pytest.raises(CaptureError, match="没有数据行"):
        EvaAdapter().parse_capture(capture_for(body, capture_data), body)


def test_eva_adapter_rejects_invalid_row(capture_data: dict[str, Any]) -> None:
    body = FIXTURE.read_bytes().replace(b"=4.0=3.0=11.0", b"=9.0=3.0=11.0", 1)

    with pytest.raises(CaptureError, match="第 2 行无效"):
        EvaAdapter().parse_capture(capture_for(body, capture_data), body)


def test_eva_adapter_rejects_duplicate_vote(capture_data: dict[str, Any]) -> None:
    lines = FIXTURE.read_bytes().splitlines(keepends=True)
    body = b"".join([lines[0], lines[1], lines[1]])

    with pytest.raises(CaptureError, match="重复评分"):
        EvaAdapter().parse_capture(capture_for(body, capture_data), body)


def test_eva_adapter_rejects_non_utf8(capture_data: dict[str, Any]) -> None:
    body = b"\xff\xfe"

    with pytest.raises(CaptureError, match="UTF-8"):
        EvaAdapter().parse_capture(capture_for(body, capture_data), body)
