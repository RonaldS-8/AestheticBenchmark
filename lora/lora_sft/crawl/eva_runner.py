"""Download and format the public EVA research dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from lora_sft.crawl.adapters.eva import EvaAdapter
from lora_sft.crawl.dataset_schema import (
    DatasetCrawlManifest,
    DatasetImageRecord,
    DatasetScoreRecord,
)
from lora_sft.crawl.http import (
    DownloadResult,
    FetchResult,
    PoliteHttpClient,
    sha256_file,
)
from lora_sft.crawl.schema import CapturedResponse

EVA_PAGE_URL = "https://github.com/kang-gnak/eva-dataset"
EVA_README_URL = (
    "https://raw.githubusercontent.com/kang-gnak/eva-dataset/master/readme.md"
)
EVA_RATINGS_URL = (
    "https://raw.githubusercontent.com/kang-gnak/eva-dataset/"
    "master/data/votes_filtered.csv"
)
EVA_ARCHIVE_URLS = tuple(
    "https://raw.githubusercontent.com/kang-gnak/eva-dataset/"
    f"master/images/EVA_together.zip.{part:03d}"
    for part in range(1, 8)
)
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


@dataclass(frozen=True)
class EvaCrawlResult:
    """Summary returned to the CLI after a crawl."""

    record_count: int
    vote_count: int
    scores_path: Path | None
    extracted_images: int
    dry_run: bool


@dataclass(frozen=True)
class EvaExtractionResult:
    """Counts produced while selecting only images with filtered ratings."""

    extracted_count: int
    archive_image_count: int
    excluded_unrated_ids: tuple[str, ...]


def _capture(
    result: FetchResult,
    *,
    capture_id: str,
    body_path: str,
    captured_at: datetime,
) -> CapturedResponse:
    return CapturedResponse(
        schema_version="1.0",
        capture_id=capture_id,
        site="eva",
        method="GET",
        url=result.url,
        status_code=result.status_code,
        content_type=result.content_type,
        captured_at=captured_at,
        body_path=body_path,
        body_sha256=hashlib.sha256(result.body).hexdigest(),
        body_size=len(result.body),
    )


def _download_capture(
    result: DownloadResult,
    *,
    capture_id: str,
    body_path: str,
    captured_at: datetime,
) -> CapturedResponse:
    return CapturedResponse(
        schema_version="1.0",
        capture_id=capture_id,
        site="eva",
        method="GET",
        url=result.url,
        status_code=result.status_code,
        content_type=result.content_type,
        captured_at=captured_at,
        body_path=body_path,
        body_sha256=result.sha256,
        body_size=result.size,
    )


def _existing_download(url: str, path: Path) -> DownloadResult:
    """Describe one previously completed part for an explicit resume run."""

    if not path.is_file():
        raise FileNotFoundError(f"恢复下载时缺少分卷: {path}")
    return DownloadResult(
        url=url,
        status_code=200,
        content_type="application/octet-stream",
        path=path,
        sha256=sha256_file(path),
        size=path.stat().st_size,
    )


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(record.model_dump_json())  # type: ignore[attr-defined]
            output.write("\n")


def _build_image_records(
    data_root: Path, score_records: list[DatasetScoreRecord]
) -> list[DatasetImageRecord]:
    """Hash extracted JPEGs and reject duplicate or malformed source files."""

    records: list[DatasetImageRecord] = []
    digest_owners: dict[str, str] = {}
    for score in score_records:
        relative_path = f"images/eva/{score.source_image_id}.jpg"
        path = data_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"EVA 评分对应图片不存在: {path}")
        size = path.stat().st_size
        if size < 2:
            raise ValueError(f"EVA 图片文件过短: {path}")
        with path.open("rb") as source:
            start = source.read(2)
            source.seek(-2, 2)
            end = source.read(2)
        if start != b"\xff\xd8":
            raise ValueError(f"EVA 图片不是 JPEG 数据: {path}")
        digest = sha256_file(path)
        owner = digest_owners.get(digest)
        if owner is not None:
            raise ValueError(f"EVA 图片内容重复: {owner} 与 {score.source_image_id}")
        digest_owners[digest] = score.source_image_id
        records.append(
            DatasetImageRecord(
                schema_version="1.0",
                dataset="eva",
                source_image_id=score.source_image_id,
                image_path=relative_path,
                sha256=digest,
                format="jpeg",
                size_bytes=size,
                has_eoi_marker=end == b"\xff\xd9",
            )
        )
    return records


def _combine_archive(parts: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if destination.exists():
        old = sha256_file(destination)
        new = sha256_file(temporary)
        if old != new:
            temporary.unlink()
            raise ValueError(f"合并归档已存在且内容不同: {destination}")
        temporary.unlink()
    else:
        temporary.replace(destination)


def _extract_images(
    archive: Path, destination: Path, expected_ids: set[str]
) -> EvaExtractionResult:
    destination.mkdir(parents=True, exist_ok=True)
    found: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"EVA 归档包含不安全路径: {member.filename}")
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(f"EVA 归档包含非 JPEG 文件: {member.filename}")
            image_id = path.stem
            if image_id in found:
                raise ValueError(f"EVA 归档包含重复图片 id: {image_id}")
            found[image_id] = member
        missing_ids = expected_ids - found.keys()
        if missing_ids:
            missing = sorted(missing_ids, key=int)[:5]
            raise ValueError(f"EVA 图片包缺少评分表中的图片: missing={missing}")
        for image_id in sorted(expected_ids, key=int):
            member = found[image_id]
            target = destination / f"{image_id}.jpg"
            with source.open(member) as input_file:
                body = input_file.read()
            if target.exists():
                if (
                    hashlib.sha256(target.read_bytes()).digest()
                    != hashlib.sha256(body).digest()
                ):
                    raise ValueError(f"目标图片已存在且内容不同: {target}")
                continue
            target.write_bytes(body)
    excluded = tuple(sorted(found.keys() - expected_ids, key=int))
    return EvaExtractionResult(len(expected_ids), len(found), excluded)


def crawl_eva(
    data_root: Path,
    *,
    dry_run: bool,
    download_images: bool,
    reuse_existing_images: bool = False,
    client: PoliteHttpClient | None = None,
    now: Callable[[], datetime] | None = None,
) -> EvaCrawlResult:
    """Fetch EVA ratings, validate them and optionally download all images."""

    http = client or PoliteHttpClient()
    clock = now or (lambda: datetime.now(SHANGHAI_TIMEZONE))
    started_at = clock()
    run_id = started_at.strftime("%Y%m%d-%H%M%S")

    readme = http.get_bytes(EVA_README_URL)
    ratings = http.get_bytes(EVA_RATINGS_URL)
    readme_capture = _capture(
        readme,
        capture_id=f"eva-{run_id}-readme",
        body_path="raw/eva/readme.md",
        captured_at=started_at,
    )
    ratings_capture = _capture(
        ratings,
        capture_id=f"eva-{run_id}-ratings",
        body_path="raw/eva/votes_filtered.csv",
        captured_at=started_at,
    )
    score_records = EvaAdapter().parse_capture(ratings_capture, ratings.body)
    vote_count = sum(record.aesthetic.count for record in score_records)
    if dry_run:
        return EvaCrawlResult(len(score_records), vote_count, None, 0, True)

    raw_root = data_root / "raw" / "eva"
    manifest_root = data_root / "manifests" / "eva"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "readme.md").write_bytes(readme.body)
    (raw_root / "votes_filtered.csv").write_bytes(ratings.body)

    captures: list[CapturedResponse] = [readme_capture, ratings_capture]
    extracted = 0
    archive_image_count: int | None = None
    excluded_unrated_ids: tuple[str, ...] | None = None
    image_records: list[DatasetImageRecord] | None = None
    if download_images:
        part_paths: list[Path] = []
        for index, url in enumerate(EVA_ARCHIVE_URLS, start=1):
            part_path = raw_root / "images" / f"EVA_together.zip.{index:03d}"
            result = (
                _existing_download(url, part_path)
                if reuse_existing_images
                else http.download(url, part_path)
            )
            part_paths.append(part_path)
            captures.append(
                _download_capture(
                    result,
                    capture_id=f"eva-{run_id}-images-{index:03d}",
                    body_path=f"raw/eva/images/{part_path.name}",
                    captured_at=(
                        datetime.fromtimestamp(
                            part_path.stat().st_mtime, tz=SHANGHAI_TIMEZONE
                        )
                        if reuse_existing_images
                        else clock()
                    ),
                )
            )
        archive = raw_root / "EVA_together.zip"
        _combine_archive(part_paths, archive)
        extraction = _extract_images(
            archive,
            data_root / "images" / "eva",
            {record.source_image_id for record in score_records},
        )
        extracted = extraction.extracted_count
        archive_image_count = extraction.archive_image_count
        excluded_unrated_ids = extraction.excluded_unrated_ids
        image_records = _build_image_records(data_root, score_records)

    scores_path = manifest_root / "scores.jsonl"
    captures_path = manifest_root / "captures.jsonl"
    _write_jsonl(scores_path, list(score_records))
    _write_jsonl(captures_path, list(captures))
    images_path: str | None = None
    if image_records is not None:
        images_path = "manifests/eva/images.jsonl"
        _write_jsonl(data_root / images_path, list(image_records))
    excluded_path: str | None = None
    if excluded_unrated_ids is not None:
        excluded_path = "manifests/eva/excluded_unrated.json"
        (data_root / excluded_path).write_text(
            json.dumps(
                {
                    "dataset": "eva",
                    "reason": "image has no entry in votes_filtered.csv",
                    "count": len(excluded_unrated_ids),
                    "source_image_ids": excluded_unrated_ids,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    manifest = DatasetCrawlManifest(
        schema_version="1.0",
        dataset="eva",
        source_page_url=EVA_PAGE_URL,
        ratings_url=EVA_RATINGS_URL,
        archive_urls=EVA_ARCHIVE_URLS,
        ratings_capture_id=ratings_capture.capture_id,
        scores_path="manifests/eva/scores.jsonl",
        record_count=len(score_records),
        research_only=True,
        archive_image_count=archive_image_count,
        excluded_unrated_count=(
            len(excluded_unrated_ids) if excluded_unrated_ids is not None else None
        ),
        excluded_unrated_path=excluded_path,
        images_path=images_path,
        image_record_count=(len(image_records) if image_records is not None else None),
        jpeg_without_eoi_count=(
            sum(not record.has_eoi_marker for record in image_records)
            if image_records is not None
            else None
        ),
    )
    (manifest_root / "crawl.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return EvaCrawlResult(len(score_records), vote_count, scores_path, extracted, False)
