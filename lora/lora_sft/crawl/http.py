"""Small sequential HTTP client for public research dataset downloads."""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "Lora-SFT-EVA-research/1.0 (local dataset audit)"


def sha256_file(path: Path) -> str:
    """Return a file digest without loading the entire file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class HttpFetchError(RuntimeError):
    """Raised when a public resource cannot be fetched safely."""


class HttpAccessError(HttpFetchError):
    """Raised when access is denied or a challenge page is returned."""


@dataclass(frozen=True)
class FetchResult:
    """One completed in-memory HTTP response."""

    url: str
    status_code: int
    content_type: str
    body: bytes


@dataclass(frozen=True)
class DownloadResult:
    """Metadata for one completed file download."""

    url: str
    status_code: int
    content_type: str
    path: Path
    sha256: str
    size: int


class PoliteHttpClient:
    """Sequential HTTPS client with pacing and bounded retry/backoff."""

    def __init__(
        self,
        *,
        delay_seconds: float = 1.0,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        max_requests: int = 20,
        sleep: Callable[[float], None] = time.sleep,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        if delay_seconds < 0 or timeout_seconds <= 0:
            raise ValueError("请求间隔不能为负，超时必须大于 0")
        if max_retries < 0 or max_requests <= 0:
            raise ValueError("重试次数不能为负，请求上限必须大于 0")
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_requests = max_requests
        self._sleep = sleep
        self._opener = opener or urllib.request.build_opener()
        self.requests = 0

    def _open(self, url: str) -> BinaryIO:
        if urllib.parse.urlsplit(url).scheme != "https":
            raise HttpFetchError(f"只允许 HTTPS 下载: {url}")
        if self.requests >= self.max_requests:
            raise HttpAccessError(f"已达到单次请求上限 {self.max_requests}")
        if self.requests and self.delay_seconds:
            self._sleep(self.delay_seconds)
        self.requests += 1
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(self.max_retries + 1):
            try:
                return self._opener.open(request, timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempt < self.max_retries:
                    self._sleep(2**attempt)
                    continue
                if exc.code in (403, 429):
                    raise HttpAccessError(f"HTTP {exc.code}，访问受限: {url}") from exc
                raise HttpFetchError(f"HTTP {exc.code}: {url}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    self._sleep(2**attempt)
                    continue
                raise HttpFetchError(f"网络错误: {url}: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _metadata(response: BinaryIO, fallback_url: str) -> tuple[str, int, str]:
        final_url = response.geturl()
        if urllib.parse.urlsplit(final_url).scheme != "https":
            raise HttpAccessError(f"下载被重定向到非 HTTPS 地址: {final_url}")
        status = getattr(response, "status", 200)
        content_type = response.headers.get_content_type()
        return final_url or fallback_url, status, content_type

    @staticmethod
    def _reject_challenge(content_type: str, body: bytes, url: str) -> None:
        if content_type != "text/html":
            return
        lowered = body[:200_000].lower()
        if b"captcha" in lowered or b"verify you are human" in lowered:
            raise HttpAccessError(f"触发验证页，停止采集: {url}")

    def get_bytes(self, url: str) -> FetchResult:
        """Fetch one small public resource into memory."""

        with self._open(url) as response:
            final_url, status, content_type = self._metadata(response, url)
            body = response.read()
        self._reject_challenge(content_type, body, final_url)
        return FetchResult(final_url, status, content_type, body)

    def download(self, url: str, destination: Path) -> DownloadResult:
        """Stream one public resource to a checked, atomic destination."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with self._open(url) as response, temporary.open("wb") as output:
                final_url, status, content_type = self._metadata(response, url)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if content_type == "text/html":
                body = temporary.read_bytes()
                self._reject_challenge(content_type, body, final_url)
            if destination.exists():
                existing = sha256_file(destination)
                if existing != digest.hexdigest():
                    raise HttpFetchError(f"目标文件已存在且内容不同: {destination}")
                temporary.unlink()
            else:
                temporary.replace(destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        return DownloadResult(
            final_url,
            status,
            content_type,
            destination,
            digest.hexdigest(),
            size,
        )
