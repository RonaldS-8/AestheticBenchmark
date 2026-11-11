from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ptune import download
from ptune.download import (
    RECORD_FILE,
    DownloadError,
    check_model_dir,
    download_model,
    file_url,
    http_fetch,
    model_dir,
)
from ptune.runner import load_job
from ptune.schema import JobConfig, WeightFile

from tests.conftest import NOW, JobFiles

FILES = {
    "chat_template.jinja": b"{{ messages }}",
    "model.safetensors": b"weights" * 100,
    "sub/vocab.json": b"{}",
}


def weight(path: str, content: bytes) -> WeightFile:
    return WeightFile(
        path=path, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )


def job_config(job_files: JobFiles, files: dict[str, bytes] = FILES) -> JobConfig:
    config = load_job(job_files.root, job_files.job).config
    weights = tuple(weight(p, c) for p, c in files.items())
    template = hashlib.sha256(files["chat_template.jinja"]).hexdigest()
    return JobConfig.model_validate(
        config.model_dump() | {"weights": weights, "chat_template_sha256": template}
    )


class FakeSource:
    """Serves file contents by URL suffix; can serve a wrong or partial file."""

    def __init__(self, files: dict[str, bytes] = FILES) -> None:
        self.files = files
        self.urls: list[str] = []

    def __call__(self, url: str, destination: Path) -> None:
        self.urls.append(url)
        name = next(p for p in self.files if url.endswith("/" + p))
        destination.write_bytes(self.files[name])


def test_download_verifies_every_file_then_only_checks(job_files: JobFiles) -> None:
    config = job_config(job_files)
    source = FakeSource()

    first = download_model(config, job_files.root, "hf", now=NOW, fetch=source)
    second = download_model(config, job_files.root, "hf", now=NOW, fetch=source)

    assert first == second == model_dir(job_files.root, config.model)
    assert first.name == "Qwen--Qwen3.5-4B"
    assert len(source.urls) == 3
    assert (first / "sub" / "vocab.json").read_bytes() == b"{}"
    assert check_model_dir(config, job_files.root) == first


def test_download_skips_files_already_complete(job_files: JobFiles) -> None:
    config = job_config(job_files)
    folder = model_dir(job_files.root, config.model)
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(FILES["model.safetensors"])
    source = FakeSource()

    download_model(config, job_files.root, "modelscope", now=NOW, fetch=source)

    assert not any(url.endswith("model.safetensors") for url in source.urls)


def test_download_rejects_wrong_content(job_files: JobFiles) -> None:
    config = job_config(job_files)
    wrong = FakeSource(FILES | {"model.safetensors": b"other weights"})

    with pytest.raises(DownloadError, match="不是锁定的文件"):
        download_model(config, job_files.root, "modelscope", now=NOW, fetch=wrong)
    folder = model_dir(job_files.root, config.model)
    assert not (folder / RECORD_FILE).exists()
    assert not (folder / "model.safetensors.part").exists()


def test_check_refuses_missing_other_or_changed_weights(job_files: JobFiles) -> None:
    config = job_config(job_files)
    with pytest.raises(DownloadError, match="先运行 download_model.py"):
        check_model_dir(config, job_files.root)

    folder = download_model(config, job_files.root, "hf", now=NOW, fetch=FakeSource())
    other = job_config(job_files, FILES | {"model.safetensors": b"v2"})
    with pytest.raises(DownloadError, match="下载记录与 job"):
        check_model_dir(other, job_files.root)
    (folder / "model.safetensors").write_bytes(b"truncated")
    with pytest.raises(DownloadError, match="大小不对"):
        check_model_dir(config, job_files.root)


def test_file_urls(job_files: JobFiles, monkeypatch: pytest.MonkeyPatch) -> None:
    model = job_config(job_files).model
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert file_url("hf", model, "a b.json") == (
        f"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{'a' * 40}/a%20b.json"
    )
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert file_url("hf", model, "config.json").startswith(
        "https://hf-mirror.com/Qwen/Qwen3.5-4B/resolve/"
    )
    assert file_url("modelscope", model, "config.json") == (
        "https://www.modelscope.cn/models/Qwen/Qwen3.5-4B/resolve/master/config.json"
    )


def test_job_weights_must_include_matching_template(job_files: JobFiles) -> None:
    data = job_config(job_files).model_dump()

    with pytest.raises(ValueError, match="必须包含 chat_template.jinja"):
        JobConfig.model_validate(
            data | {"weights": [weight("model.safetensors", b"x")]}
        )
    with pytest.raises(ValueError, match="哈希与配置不符"):
        JobConfig.model_validate(data | {"chat_template_sha256": "0" * 64})
    with pytest.raises(ValueError, match="重复文件"):
        JobConfig.model_validate(data | {"weights": [*data["weights"]] * 2})


# --- http_fetch, with urlopen replaced ---


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int) -> None:
        super().__init__(body)
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def fake_urlopen(
    outcomes: list[Any], seen: list[dict[str, str]]
) -> Callable[[urllib.request.Request, int], FakeResponse]:
    def urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
        seen.append(dict(request.header_items()))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return urlopen


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", {}, None)  # type: ignore[arg-type]


def test_http_fetch_resumes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "f.part"
    part.write_bytes(b"abc")
    seen: list[dict[str, str]] = []
    outcomes: list[Any] = [FakeResponse(b"def", 206)]
    monkeypatch.setattr(
        download.urllib.request, "urlopen", fake_urlopen(outcomes, seen)
    )

    http_fetch("https://x/f", part)

    assert part.read_bytes() == b"abcdef"
    assert seen[0]["Range"] == "bytes=3-"


def test_http_fetch_restarts_when_range_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "f.part"
    part.write_bytes(b"abc")
    outcomes: list[Any] = [FakeResponse(b"whole", 200)]
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(outcomes, []))

    http_fetch("https://x/f", part)

    assert part.read_bytes() == b"whole"


def test_http_fetch_retries_then_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    part = tmp_path / "f.part"
    flaky: list[Any] = [urllib.error.URLError("reset"), FakeResponse(b"ok", 200)]
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(flaky, []))
    http_fetch("https://x/f", part)
    assert part.read_bytes() == b"ok"

    down: list[Any] = [http_error(503)] * 3
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(down, []))
    with pytest.raises(DownloadError, match="HTTP 503"):
        http_fetch("https://x/f", tmp_path / "g.part")

    missing: list[Any] = [http_error(404)]
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(missing, []))
    with pytest.raises(DownloadError, match="HTTP 404"):
        http_fetch("https://x/f", tmp_path / "h.part")

    offline: list[Any] = [TimeoutError("slow")] * 3
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(offline, []))
    with pytest.raises(DownloadError, match="slow"):
        http_fetch("https://x/f", tmp_path / "i.part")


def test_http_fetch_treats_416_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "f.part"
    part.write_bytes(b"done")
    outcomes: list[Any] = [http_error(416)]
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(outcomes, []))

    http_fetch("https://x/f", part)

    assert part.read_bytes() == b"done"
