"""Download the pinned model weights on the cloud machine (no upload needed).

``data/job.json`` lists every model file with its SHA-256, taken when packing
from a copy already checked against the pinned revision. Files are fetched
one by one from the source chosen explicitly (no automatic switching):

- ``hf``: Hugging Face at the pinned revision; set ``HF_ENDPOINT`` (e.g.
  ``https://hf-mirror.com``) when huggingface.co is unreachable;
- ``modelscope``: modelscope.cn, which serves the same Qwen files but under
  its own revisions, so only the hashes prove they are the pinned ones.

Every file must match its hash or the download fails; a partial file resumes
on the next run. Files land in ``models/<owner>--<name>/`` with a
``DOWNLOAD.json`` record; runs load only from that folder.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import AwareDatetime

from ptune.schema import BaseModelRef, JobConfig, Record, WeightFile

RECORD_FILE = "DOWNLOAD.json"
CHUNK_BYTES = 1 << 20
ATTEMPTS = 3
Source = Literal["hf", "modelscope"]
# url, destination file (appended to when it already holds a prefix)
Fetch = Callable[[str, Path], None]


class DownloadError(RuntimeError):
    """Raised when the weights are missing, partial or not the pinned ones."""


class DownloadRecord(Record):
    """What was downloaded into a model folder, from where, and when."""

    model: BaseModelRef
    source: Source
    weights: tuple[WeightFile, ...]
    downloaded_at: AwareDatetime


def model_dir(root: Path, model: BaseModelRef) -> Path:
    """Where the job keeps this model's weights."""

    return root / "models" / model.model_id.replace("/", "--")


def file_url(source: Source, model: BaseModelRef, path: str) -> str:
    """Download URL of one model file."""

    name = quote(path)
    if source == "hf":
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        return (
            f"{endpoint.rstrip('/')}/{model.model_id}/resolve/{model.revision}/{name}"
        )
    return f"https://www.modelscope.cn/models/{model.model_id}/resolve/master/{name}"


def http_fetch(url: str, destination: Path) -> None:
    """Stream ``url`` into ``destination``, resuming from its current size."""

    for attempt in range(1, ATTEMPTS + 1):
        offset = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "ptune"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with destination.open(mode) as f:
                    while chunk := response.read(CHUNK_BYTES):
                        f.write(chunk)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # already complete
                return
            if exc.code < 500 or attempt == ATTEMPTS:
                raise DownloadError(f"下载失败 {url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == ATTEMPTS:
                raise DownloadError(f"下载失败 {url}: {exc}") from exc
        time.sleep(5 * attempt)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, weight: WeightFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == weight.size_bytes
        and _sha256(path) == weight.sha256
    )


def check_model_dir(config: JobConfig, root: Path) -> Path:
    """Return the model folder if its record and file sizes match the job.

    Hashes were verified at download time; here only sizes are re-checked,
    so a run does not re-read gigabytes of weights.
    """

    folder = model_dir(root, config.model)
    record_path = folder / RECORD_FILE
    if not record_path.is_file():
        raise DownloadError(f"{folder} 没有下载完成的记录，先运行 download_model.py")
    record = DownloadRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    if record.model != config.model or record.weights != config.weights:
        raise DownloadError(f"{folder} 的下载记录与 job 要求的模型文件不符")
    for weight in config.weights:
        path = folder / weight.path
        if not path.is_file() or path.stat().st_size != weight.size_bytes:
            raise DownloadError(f"{path} 缺失或大小不对，重新运行 download_model.py")
    return folder


def download_model(
    config: JobConfig,
    root: Path,
    source: Source,
    *,
    now: datetime,
    fetch: Fetch = http_fetch,
) -> Path:
    """Fetch every missing file, verify all hashes, then write the record."""

    folder = model_dir(root, config.model)
    if (folder / RECORD_FILE).is_file():
        return check_model_dir(config, root)
    for weight in config.weights:
        path = folder / weight.path
        if _matches(path, weight):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        fetch(file_url(source, config.model, weight.path), part)
        if not _matches(part, weight):
            part.unlink()
            raise DownloadError(
                f"{weight.path} 的大小或 sha256 与 job 不符（来源 {source}），"
                "已删除，不是锁定的文件"
            )
        part.replace(path)
    record = DownloadRecord(
        model=config.model, source=source, weights=config.weights, downloaded_at=now
    )
    (folder / RECORD_FILE).write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return folder
