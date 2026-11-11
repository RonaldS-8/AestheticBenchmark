"""ImageNet CNN backbones with multi-level pooled features.

Each backbone runs on the image at its stored size (no resize, no crop, aspect
ratio kept), normalized with ImageNet statistics. Forward hooks take the
output of four stages from fine to coarse; each is global-average-pooled and
returned separately, so a probe can use all levels or only the last.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models as tvm
from torchvision.transforms.functional import pil_to_tensor

from aes_head.data import sha256_file

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BackboneError(ValueError):
    """Raised for an unknown backbone or a checkpoint that does not match."""


@dataclass(frozen=True)
class BackboneSpec:
    """How to build a backbone, where its weights live and which stages to tap."""

    name: str
    build: Callable[[], nn.Module]
    weights_url: str
    taps: tuple[str, ...]
    dims: tuple[int, ...]


SPECS: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        name="resnet50",
        build=lambda: tvm.resnet50(weights=None),
        weights_url=tvm.ResNet50_Weights.IMAGENET1K_V2.url,
        taps=("layer1", "layer2", "layer3", "layer4"),
        dims=(256, 512, 1024, 2048),
    ),
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        build=lambda: tvm.efficientnet_b0(weights=None),
        weights_url=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1.url,
        # Strides 4, 8, 16 and the final 1x1 conv at stride 32.
        taps=("features.2", "features.3", "features.5", "features.8"),
        dims=(24, 40, 112, 1280),
    ),
    "convnext_tiny": BackboneSpec(
        name="convnext_tiny",
        build=lambda: tvm.convnext_tiny(weights=None),
        weights_url=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,
        taps=("features.1", "features.3", "features.5", "features.7"),
        dims=(96, 192, 384, 768),
    ),
}
BACKBONES = tuple(SPECS)


def get_spec(name: str) -> BackboneSpec:
    """The spec of a known backbone."""

    if name not in SPECS:
        raise BackboneError(f"未知骨干 {name!r}，可选 {list(SPECS)}")
    return SPECS[name]


def build_backbone(name: str) -> nn.Module:
    """A randomly initialized backbone (load weights with ``load_backbone``)."""

    return get_spec(name).build()


def load_backbone(name: str, weights_path: Path, sha256: str) -> nn.Module:
    """Build a backbone and load its checkpoint after checking the file hash."""

    if not weights_path.is_file():
        raise BackboneError(f"权重文件不存在: {weights_path}")
    if sha256_file(weights_path) != sha256:
        raise BackboneError(f"权重文件哈希与 WEIGHTS.json 不符: {weights_path}")
    model = build_backbone(name)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB image to a normalized ``[1, 3, H, W]`` float tensor at native size."""

    pixels = pil_to_tensor(image).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((pixels - mean) / std).unsqueeze(0)


class MultiLevelExtractor(nn.Module):
    """Run a classifier backbone and average-pool the tapped stage outputs."""

    def __init__(self, spec: BackboneSpec, model: nn.Module) -> None:
        super().__init__()
        self.spec = spec
        self.model = model.eval()
        self._pooled: dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        missing = [tap for tap in spec.taps if tap not in modules]
        if missing:
            raise BackboneError(f"{spec.name} 没有这些层: {missing}")
        for tap in spec.taps:
            modules[tap].register_forward_hook(self._hook(tap))

    def _hook(self, tap: str) -> Callable[[nn.Module, object, torch.Tensor], None]:
        def store(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
            self._pooled[tap] = output.mean(dim=(2, 3))

        return store

    def forward(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        """Pooled features per tapped stage, each ``[batch, channels]``."""

        self._pooled.clear()
        self.model(pixels)
        levels = [self._pooled[tap] for tap in self.spec.taps]
        dims = tuple(level.shape[1] for level in levels)
        if dims != self.spec.dims:
            raise BackboneError(f"{self.spec.name} 各层通道数 {dims} 与预期不符")
        return levels
