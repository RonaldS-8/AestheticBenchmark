from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from lora_sft.crawl.base import CaptureError, SiteAdapter
from lora_sft.crawl.schema import CapturedResponse, PhotoCandidate


class FakeSiteAdapter(SiteAdapter):
    site = "example_site"

    def _parse_body(
        self, capture: CapturedResponse, body: bytes
    ) -> list[PhotoCandidate]:
        return [
            PhotoCandidate.model_validate({**item, "capture_id": capture.capture_id})
            for item in json.loads(body)
        ]


class WrongCaptureIdAdapter(FakeSiteAdapter):
    def _parse_body(
        self, capture: CapturedResponse, body: bytes
    ) -> list[PhotoCandidate]:
        return [
            PhotoCandidate.model_validate({**item, "capture_id": "other-capture"})
            for item in json.loads(body)
        ]


def make_capture(
    capture_data: dict[str, Any], body: bytes, **overrides: object
) -> CapturedResponse:
    capture_data.update(
        body_sha256=hashlib.sha256(body).hexdigest(), body_size=len(body)
    )
    capture_data.update(overrides)
    return CapturedResponse.model_validate(capture_data)


def make_body(*candidates: dict[str, Any]) -> bytes:
    items = [{k: v for k, v in c.items() if k != "capture_id"} for c in candidates]
    return json.dumps(items).encode("utf-8")


def test_parse_capture_returns_candidates(
    capture_data: dict[str, Any], candidate_data: dict[str, Any]
) -> None:
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body)

    candidates = FakeSiteAdapter().parse_capture(capture, body)

    assert len(candidates) == 1
    assert candidates[0].capture_id == capture.capture_id
    assert candidates[0].source.site_photo_id == "889123"


def test_empty_listing_returns_no_candidates(capture_data: dict[str, Any]) -> None:
    body = make_body()
    capture = make_capture(capture_data, body)

    assert FakeSiteAdapter().parse_capture(capture, body) == []


def test_capture_from_other_site_rejected(
    capture_data: dict[str, Any], candidate_data: dict[str, Any]
) -> None:
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body, site="other_site")

    with pytest.raises(CaptureError, match="不是 example_site"):
        FakeSiteAdapter().parse_capture(capture, body)


@pytest.mark.parametrize("status", [199, 300, 403, 500])
def test_non_success_status_rejected(
    capture_data: dict[str, Any], candidate_data: dict[str, Any], status: int
) -> None:
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body, status_code=status)

    with pytest.raises(CaptureError, match=f"状态码 {status}"):
        FakeSiteAdapter().parse_capture(capture, body)


@pytest.mark.parametrize("status", [200, 299])
def test_success_status_boundary_accepted(
    capture_data: dict[str, Any], candidate_data: dict[str, Any], status: int
) -> None:
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body, status_code=status)

    assert len(FakeSiteAdapter().parse_capture(capture, body)) == 1


def test_tampered_body_rejected(
    capture_data: dict[str, Any], candidate_data: dict[str, Any]
) -> None:
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body)

    with pytest.raises(CaptureError, match="sha256"):
        FakeSiteAdapter().parse_capture(capture, body + b" ")


def test_candidate_with_wrong_capture_id_rejected(
    capture_data: dict[str, Any], candidate_data: dict[str, Any]
) -> None:
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body)

    with pytest.raises(CaptureError, match="与 capture 不对应"):
        WrongCaptureIdAdapter().parse_capture(capture, body)


def test_candidate_from_other_site_rejected(
    capture_data: dict[str, Any], candidate_data: dict[str, Any]
) -> None:
    candidate_data["source"]["site"] = "other_site"
    body = make_body(candidate_data)
    capture = make_capture(capture_data, body)

    with pytest.raises(CaptureError, match="与 capture 不对应"):
        FakeSiteAdapter().parse_capture(capture, body)


def test_base_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        SiteAdapter()  # type: ignore[abstract]
