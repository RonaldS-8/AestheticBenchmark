from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from lora_sft.crawl.adapters.eva import EvaAdapter
from lora_sft.crawl.eva_runner import (
    EVA_RATINGS_URL,
    EVA_README_URL,
    _build_image_records,
    _combine_archive,
    _extract_images,
    crawl_eva,
)
from lora_sft.crawl.http import FetchResult
from lora_sft.crawl.schema import CapturedResponse

FIXTURE = Path(__file__).parent / "fixtures" / "eva_votes_filtered.csv"


class FakeClient:
    def get_bytes(self, url: str) -> FetchResult:
        if url == EVA_README_URL:
            body = b"EVA dataset"
            content_type = "text/markdown"
        elif url == EVA_RATINGS_URL:
            body = FIXTURE.read_bytes()
            content_type = "text/csv"
        else:  # pragma: no cover - test fake contract
            raise AssertionError(url)
        return FetchResult(url, 200, content_type, body)


NOW = datetime.fromisoformat("2026-11-02T16:00:00+08:00")


def _capture_for_fixture() -> CapturedResponse:
    body = FIXTURE.read_bytes()
    return CapturedResponse(
        schema_version="1.0",
        capture_id="eva-test-ratings",
        site="eva",
        method="GET",
        url=EVA_RATINGS_URL,
        status_code=200,
        content_type="text/csv",
        captured_at=NOW,
        body_path="raw/eva/votes_filtered.csv",
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_size=len(body),
    )


def test_crawl_eva_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = crawl_eva(
        tmp_path,
        dry_run=True,
        download_images=False,
        client=FakeClient(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert result.record_count == 2
    assert result.vote_count == 3
    assert result.scores_path is None
    assert not list(tmp_path.rglob("*"))


def test_crawl_eva_writes_provenance_and_scores(tmp_path: Path) -> None:
    result = crawl_eva(
        tmp_path,
        dry_run=False,
        download_images=False,
        client=FakeClient(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert result.scores_path == tmp_path / "manifests" / "eva" / "scores.jsonl"
    scores = result.scores_path.read_text(encoding="utf-8").splitlines()
    assert len(scores) == 2
    assert json.loads(scores[0])["source_image_id"] == "71"
    captures = (tmp_path / "manifests" / "eva" / "captures.jsonl").read_text(
        encoding="utf-8"
    )
    assert "eva-20261102-160000-ratings" in captures
    manifest = json.loads(
        (tmp_path / "manifests" / "eva" / "crawl.json").read_text(encoding="utf-8")
    )
    assert manifest["record_count"] == 2
    assert manifest["research_only"] is True
    assert manifest["archive_image_count"] is None
    assert manifest["excluded_unrated_count"] is None
    assert manifest["excluded_unrated_path"] is None
    assert manifest["images_path"] is None
    assert manifest["image_record_count"] is None
    assert manifest["jpeg_without_eoi_count"] is None


def make_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return output.getvalue()


def test_combine_and_extract_images(tmp_path: Path) -> None:
    archive_body = make_zip({"nested/71.jpg": b"one", "106.jpeg": b"two"})
    midpoint = len(archive_body) // 2
    parts = [tmp_path / "one", tmp_path / "two"]
    parts[0].write_bytes(archive_body[:midpoint])
    parts[1].write_bytes(archive_body[midpoint:])
    archive = tmp_path / "combined.zip"

    _combine_archive(parts, archive)
    extraction = _extract_images(archive, tmp_path / "images", {"71", "106"})

    assert extraction.extracted_count == 2
    assert extraction.archive_image_count == 2
    assert extraction.excluded_unrated_ids == ()
    assert (tmp_path / "images" / "71.jpg").read_bytes() == b"one"
    assert (tmp_path / "images" / "106.jpg").read_bytes() == b"two"
    _combine_archive(parts, archive)
    assert _extract_images(archive, tmp_path / "images", {"71", "106"}) == extraction


def test_extract_images_excludes_archive_members_without_ratings(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "extra.zip"
    archive.write_bytes(
        make_zip({"71.jpg": b"one", "106.jpg": b"two", "580.jpg": b"extra"})
    )

    extraction = _extract_images(archive, tmp_path / "images", {"71", "106"})

    assert extraction.extracted_count == 2
    assert extraction.archive_image_count == 3
    assert extraction.excluded_unrated_ids == ("580",)
    assert not (tmp_path / "images" / "580.jpg").exists()


def test_build_image_records_hashes_images_and_marks_missing_eoi(
    tmp_path: Path,
) -> None:
    parsed = EvaAdapter().parse_capture(_capture_for_fixture(), FIXTURE.read_bytes())
    image_root = tmp_path / "images" / "eva"
    image_root.mkdir(parents=True)
    (image_root / "71.jpg").write_bytes(b"\xff\xd8one\xff\xd9")
    (image_root / "106.jpg").write_bytes(b"\xff\xd8two")

    records = _build_image_records(tmp_path, parsed)

    assert len(records) == 2
    assert records[0].image_path == "images/eva/71.jpg"
    assert records[0].has_eoi_marker is True
    assert records[1].has_eoi_marker is False


@pytest.mark.parametrize(
    ("files", "expected", "message"),
    [
        ({"../71.jpg": b"x"}, {"71"}, "不安全路径"),
        ({"71.png": b"x"}, {"71"}, "非 JPEG"),
        ({"a/71.jpg": b"x", "b/71.jpeg": b"y"}, {"71"}, "重复图片"),
        ({"71.jpg": b"x"}, {"71", "106"}, "缺少评分表中的图片"),
    ],
)
def test_extract_images_rejects_bad_archive(
    tmp_path: Path, files: dict[str, bytes], expected: set[str], message: str
) -> None:
    archive = tmp_path / "bad.zip"
    archive.write_bytes(make_zip(files))

    with pytest.raises(ValueError, match=message):
        _extract_images(archive, tmp_path / "images", expected)
