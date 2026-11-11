from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from aes_head import weights
from aes_head.backbones import get_spec
from aes_head.weights import (
    WeightsError,
    download_weights,
    hash_prefix,
    load_index,
)

CONTENT = b"fake checkpoint"
PREFIX = hashlib.sha256(CONTENT).hexdigest()[:8]
URL = f"https://example.test/models/fake-{PREFIX}.pth"


class FakeDownloader:
    """Writes fixed bytes instead of downloading; counts calls."""

    def __init__(self, content: bytes = CONTENT) -> None:
        self.content = content
        self.calls: list[str] = []

    def __call__(self, url: str, dst: str, **_kwargs: object) -> None:
        self.calls.append(url)
        Path(dst).write_bytes(self.content)


@pytest.fixture(autouse=True)
def fake_url(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = replace(get_spec("efficientnet_b0"), weights_url=URL)
    monkeypatch.setattr(weights, "get_spec", lambda _name: spec)


def test_hash_prefix_comes_from_the_file_name() -> None:
    assert hash_prefix("https://x.test/resnet50-11ad3fa6.pth") == "11ad3fa6"
    with pytest.raises(WeightsError, match="哈希前缀"):
        hash_prefix("https://x.test/resnet50.pth")


def test_download_writes_index_and_skips_existing(tmp_path: Path) -> None:
    downloader = FakeDownloader()

    index = download_weights(tmp_path, ["efficientnet_b0"], downloader)
    again = download_weights(tmp_path, ["efficientnet_b0"], downloader)

    assert downloader.calls == [URL]
    assert index == again == load_index(tmp_path)
    entry = index.find("efficientnet_b0")
    assert entry.sha256 == hashlib.sha256(CONTENT).hexdigest()
    assert entry.size_bytes == len(CONTENT)


def test_download_rejects_wrong_file(tmp_path: Path) -> None:
    with pytest.raises(WeightsError, match="哈希不以"):
        download_weights(tmp_path, ["efficientnet_b0"], FakeDownloader(b"other"))


def test_download_rejects_index_conflict(tmp_path: Path) -> None:
    download_weights(tmp_path, ["efficientnet_b0"], FakeDownloader())
    index_path = tmp_path / "WEIGHTS.json"
    text = index_path.read_text(encoding="utf-8")
    index_path.write_text(text.replace('"size_bytes": 15', '"size_bytes": 16'))

    with pytest.raises(WeightsError, match="记录的不同"):
        download_weights(tmp_path, ["efficientnet_b0"], FakeDownloader())


def test_load_index_requires_file(tmp_path: Path) -> None:
    with pytest.raises(WeightsError, match="download_weights"):
        load_index(tmp_path)
