from __future__ import annotations

import hashlib
import io
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from lora_sft.crawl.http import (
    HttpAccessError,
    HttpFetchError,
    PoliteHttpClient,
    sha256_file,
)


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "https://example.test/data",
        content_type: str = "application/octet-stream",
        status: int = 200,
    ) -> None:
        super().__init__(body)
        self._url = url
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        return self._url


class FakeOpener:
    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def open(self, request: object, *, timeout: int) -> FakeResponse:
        del request, timeout
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/data", code, "test", Message(), None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"delay_seconds": -1},
        {"timeout_seconds": 0},
        {"max_retries": -1},
        {"max_requests": 0},
    ],
)
def test_http_client_rejects_invalid_settings(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        PoliteHttpClient(**changes)


def test_get_bytes_uses_https_metadata_and_pacing() -> None:
    sleeps: list[float] = []
    opener = FakeOpener(
        FakeResponse(b"one", content_type="text/plain"),
        FakeResponse(b"two", content_type="text/csv"),
    )
    client = PoliteHttpClient(
        delay_seconds=0.5, sleep=sleeps.append, opener=opener  # type: ignore[arg-type]
    )

    first = client.get_bytes("https://example.test/one")
    second = client.get_bytes("https://example.test/two")

    assert first.body == b"one"
    assert first.content_type == "text/plain"
    assert second.body == b"two"
    assert client.requests == 2
    assert sleeps == [0.5]


def test_http_client_rejects_http_redirect_and_request_overflow() -> None:
    redirecting = PoliteHttpClient(
        opener=FakeOpener(  # type: ignore[arg-type]
            FakeResponse(b"x", url="http://example.test/insecure")
        )
    )
    with pytest.raises(HttpAccessError, match="非 HTTPS"):
        redirecting.get_bytes("https://example.test/data")

    limited = PoliteHttpClient(
        max_requests=1,
        delay_seconds=0,
        opener=FakeOpener(FakeResponse(b"x")),  # type: ignore[arg-type]
    )
    limited.get_bytes("https://example.test/data")
    with pytest.raises(HttpAccessError, match="请求上限"):
        limited.get_bytes("https://example.test/data")


def test_http_client_rejects_plain_http_without_opening() -> None:
    opener = FakeOpener(FakeResponse(b"unused"))
    client = PoliteHttpClient(opener=opener)  # type: ignore[arg-type]

    with pytest.raises(HttpFetchError, match="只允许 HTTPS"):
        client.get_bytes("http://example.test/data")

    assert opener.calls == 0


def test_retryable_http_error_recovers() -> None:
    sleeps: list[float] = []
    opener = FakeOpener(http_error(503), FakeResponse(b"ok"))
    client = PoliteHttpClient(
        max_retries=1,
        delay_seconds=0,
        sleep=sleeps.append,
        opener=opener,  # type: ignore[arg-type]
    )

    assert client.get_bytes("https://example.test/data").body == b"ok"
    assert sleeps == [1]
    assert opener.calls == 2


@pytest.mark.parametrize(
    ("code", "error_type", "message"),
    [
        (403, HttpAccessError, "访问受限"),
        (429, HttpAccessError, "访问受限"),
        (404, HttpFetchError, "HTTP 404"),
    ],
)
def test_http_error_is_classified(
    code: int, error_type: type[Exception], message: str
) -> None:
    client = PoliteHttpClient(
        max_retries=0,
        opener=FakeOpener(http_error(code)),  # type: ignore[arg-type]
    )

    with pytest.raises(error_type, match=message):
        client.get_bytes("https://example.test/data")


def test_network_error_retries_then_fails() -> None:
    error = urllib.error.URLError("offline")
    opener = FakeOpener(error, error)
    client = PoliteHttpClient(
        max_retries=1,
        delay_seconds=0,
        sleep=lambda _: None,
        opener=opener,  # type: ignore[arg-type]
    )

    with pytest.raises(HttpFetchError, match="网络错误"):
        client.get_bytes("https://example.test/data")


@pytest.mark.parametrize("marker", [b"captcha", b"VERIFY YOU ARE HUMAN"])
def test_challenge_page_is_rejected(marker: bytes) -> None:
    client = PoliteHttpClient(
        opener=FakeOpener(  # type: ignore[arg-type]
            FakeResponse(marker, content_type="text/html")
        )
    )

    with pytest.raises(HttpAccessError, match="验证页"):
        client.get_bytes("https://example.test/data")


def test_download_is_atomic_and_accepts_identical_existing_file(
    tmp_path: Path,
) -> None:
    body = b"downloaded"
    destination = tmp_path / "nested" / "part.bin"
    client = PoliteHttpClient(
        delay_seconds=0,
        opener=FakeOpener(  # type: ignore[arg-type]
            FakeResponse(body), FakeResponse(body)
        ),
    )

    result = client.download("https://example.test/data", destination)
    repeated = client.download("https://example.test/data", destination)

    assert destination.read_bytes() == body
    assert result.sha256 == hashlib.sha256(body).hexdigest()
    assert result.size == len(body)
    assert repeated.sha256 == sha256_file(destination)
    assert not destination.with_suffix(".bin.part").exists()


def test_download_rejects_different_existing_file_and_cleans_part(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "part.bin"
    destination.write_bytes(b"old")
    client = PoliteHttpClient(
        opener=FakeOpener(FakeResponse(b"new"))  # type: ignore[arg-type]
    )

    with pytest.raises(HttpFetchError, match="内容不同"):
        client.download("https://example.test/data", destination)

    assert destination.read_bytes() == b"old"
    assert not destination.with_suffix(".bin.part").exists()


def test_download_rejects_challenge_and_cleans_part(tmp_path: Path) -> None:
    destination = tmp_path / "part.bin"
    client = PoliteHttpClient(
        opener=FakeOpener(  # type: ignore[arg-type]
            FakeResponse(b"captcha", content_type="text/html")
        )
    )

    with pytest.raises(HttpAccessError, match="验证页"):
        client.download("https://example.test/data", destination)

    assert not destination.exists()
    assert not destination.with_suffix(".bin.part").exists()
