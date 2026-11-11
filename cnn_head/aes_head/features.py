"""Extract and cache frozen-backbone features for train and val images.

The probe never touches the eval split: asking for eval items is an error.
A cache file records the backbone, the checkpoint hash, the items hash and
the image order, and is only reused when all of them match.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from aes_head.backbones import MultiLevelExtractor, to_tensor
from aes_head.console import echo
from aes_head.data import load_rgb
from aes_head.schema import HeadItem

FORMAT_VERSION = 1


class FeatureError(ValueError):
    """Raised for eval items or a cache that does not match the request."""


@dataclass(frozen=True)
class FeatureSet:
    """Pooled features per tapped stage for a list of images."""

    backbone: str
    weights_sha256: str
    items_sha256: str
    image_ids: tuple[str, ...]
    levels: tuple[torch.Tensor, ...]
    latency_ms: tuple[float, ...]

    def matrix(self, layers: str) -> torch.Tensor:
        """``[images, features]`` float64: all stages concatenated, or the last."""

        if layers == "all":
            return torch.cat(self.levels, dim=1)
        if layers == "last":
            return self.levels[-1]
        raise FeatureError(f"layers 只能是 all 或 last，实际 {layers!r}")


def extract_features(
    data_root: Path,
    items: Sequence[HeadItem],
    extractor: MultiLevelExtractor,
    *,
    weights_sha256: str,
    items_sha256: str,
    device: str,
    log: Callable[[str], None] = echo,
) -> FeatureSet:
    """Run the backbone on every item, one image at a time at native size."""

    if any(item.split == "eval" for item in items):
        raise FeatureError("探针只用 train 和 val，不能抽 eval 集的特征")
    extractor.to(device)
    per_level: list[list[torch.Tensor]] = [[] for _ in extractor.spec.taps]
    latency: list[float] = []
    for index, item in enumerate(items, 1):
        pixels = to_tensor(load_rgb(data_root, item)).to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            levels = extractor(pixels)
        latency.append((time.perf_counter() - start) * 1000)
        for bucket, level in zip(per_level, levels, strict=True):
            bucket.append(level[0].to("cpu", torch.float64))
        if index % 200 == 0 or index == len(items):
            log(f"{extractor.spec.name}: {index}/{len(items)}")
    return FeatureSet(
        backbone=extractor.spec.name,
        weights_sha256=weights_sha256,
        items_sha256=items_sha256,
        image_ids=tuple(item.image_id for item in items),
        levels=tuple(torch.stack(bucket) for bucket in per_level),
        latency_ms=tuple(latency),
    )


def save_features(path: Path, features: FeatureSet) -> None:
    """Write a cache file; an existing file is never overwritten."""

    if path.exists():
        raise FeatureError(f"{path} 已存在，不覆盖")
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "backbone": features.backbone,
            "weights_sha256": features.weights_sha256,
            "items_sha256": features.items_sha256,
            "image_ids": list(features.image_ids),
            "levels": list(features.levels),
            "latency_ms": list(features.latency_ms),
        },
        path,
    )


def load_features(
    path: Path,
    *,
    backbone: str,
    weights_sha256: str,
    items_sha256: str,
    image_ids: Sequence[str],
) -> FeatureSet:
    """Read a cache file and check it was made for exactly this request."""

    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": FORMAT_VERSION,
        "backbone": backbone,
        "weights_sha256": weights_sha256,
        "items_sha256": items_sha256,
        "image_ids": list(image_ids),
    }
    for key, value in expected.items():
        if saved[key] != value:
            raise FeatureError(f"{path} 的 {key} 与本次运行不符，删除缓存后重跑")
    levels = tuple(saved["levels"])
    if any(level.shape[0] != len(image_ids) for level in levels):
        raise FeatureError(f"{path} 的特征行数与图片数不符")
    return FeatureSet(
        backbone=backbone,
        weights_sha256=weights_sha256,
        items_sha256=items_sha256,
        image_ids=tuple(image_ids),
        levels=levels,
        latency_ms=tuple(saved["latency_ms"]),
    )
