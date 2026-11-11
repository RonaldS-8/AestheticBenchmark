"""Base class for per-site crawler adapters."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import ClassVar

from lora_sft.crawl.schema import CapturedResponse, PhotoCandidate


class CaptureError(ValueError):
    """Raised when a capture cannot be parsed into trustworthy candidates."""


def validate_capture(
    capture: CapturedResponse, body: bytes, expected_site: str
) -> None:
    """Validate source, status and digest before an adapter parses a body."""

    if capture.site != expected_site:
        raise CaptureError(
            f"capture {capture.capture_id} 来自 {capture.site}，"
            f"不是 {expected_site}"
        )
    if not 200 <= capture.status_code < 300:
        raise CaptureError(
            f"capture {capture.capture_id} 状态码 {capture.status_code}，不能解析"
        )
    if hashlib.sha256(body).hexdigest() != capture.body_sha256:
        raise CaptureError(f"capture {capture.capture_id} 的 body 与 sha256 不符")


class SiteAdapter(ABC):
    """Turns raw captures from one site into photo candidates.

    Subclasses are named ``<Site>Adapter``, set ``site`` and implement
    ``_parse_body``. Adapters only parse: network access, pacing and
    downloads belong to the crawler runner.
    """

    site: ClassVar[str]

    def parse_capture(
        self, capture: CapturedResponse, body: bytes
    ) -> list[PhotoCandidate]:
        """Check the capture belongs to this site and is intact, then parse it."""

        validate_capture(capture, body, self.site)
        candidates = self._parse_body(capture, body)
        for candidate in candidates:
            if (
                candidate.capture_id != capture.capture_id
                or candidate.source.site != self.site
            ):
                raise CaptureError(
                    f"{type(self).__name__} 产出的候选 "
                    f"{candidate.source.site_photo_id} 与 capture 不对应"
                )
        return candidates

    @abstractmethod
    def _parse_body(
        self, capture: CapturedResponse, body: bytes
    ) -> list[PhotoCandidate]:
        """Parse one successful response body; raise on unexpected structure."""
