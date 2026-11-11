"""The frozen Qwen3.5-4B vision tower and its pooled multi-level features.

Images go through the base model's own image processor with the LoRA
config's pixel budget, so the tower sees exactly what the LoRA model sees.
One image per forward pass. Each tapped block's output tokens and the
merger's output tokens are mean-pooled in fp32 and concatenated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from qv_head.config import TowerSettings
from qv_head.weights import TOWER_DIR, TOWER_WEIGHTS_FILE, verify_tower


class TowerError(ValueError):
    """Raised for a bad runtime setting or a tower that does not fit the config."""


def check_runtime(device: str, precision: str) -> None:
    """The bf16 tower runs on CUDA only; CPU runs must say fp32."""

    if precision == "bf16" and not device.startswith("cuda"):
        raise TowerError("bf16 只在 CUDA 上用；CPU 上请用 fp32")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TowerError("CUDA 不可用")


def sync(device: str) -> None:
    """Wait for queued GPU work, so wall-clock timings are real."""

    if device.startswith("cuda"):
        torch.cuda.synchronize()


class TowerFeatures:
    """Pooled features of one image from a frozen tower."""

    def __init__(
        self,
        model: Qwen3_5VisionModel,
        processor: Callable[..., Mapping[str, torch.Tensor]],
        settings: TowerSettings,
        device: str,
    ) -> None:
        depth = len(model.blocks)
        if settings.taps[-1] >= depth:
            raise TowerError(f"taps {settings.taps} 超出视觉塔的 {depth} 个块")
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = processor
        self.settings = settings
        self.device = device
        hidden = model.config.hidden_size
        self.layer_names = tuple(f"block{i + 1}" for i in settings.taps) + ("merger",)
        self.layer_dims = (hidden,) * len(settings.taps) + (
            model.config.out_hidden_size,
        )
        self._pooled: dict[int, torch.Tensor] = {}
        for index in settings.taps:
            model.blocks[index].register_forward_hook(self._hook(index))

    def _hook(self, index: int) -> Callable[..., None]:
        def store(_module: torch.nn.Module, _args: object, output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise TowerError(f"第 {index + 1} 块的输出不是张量")
            self._pooled[index] = output.float().mean(dim=0)

        return store

    def encode(self, image: Image.Image) -> dict[str, torch.Tensor]:
        """Patchify one image with the base model's processor and pixel budget."""

        encoded = self.processor(
            images=[image],
            size={
                "shortest_edge": self.settings.min_pixels,
                "longest_edge": self.settings.max_pixels,
            },
            return_tensors="pt",
        )
        if set(encoded.keys()) != {"pixel_values", "image_grid_thw"}:
            raise TowerError(f"图像处理器输出了意外的字段: {sorted(encoded.keys())}")
        return {
            "pixel_values": encoded["pixel_values"].to(self.device),
            "image_grid_thw": encoded["image_grid_thw"].to(self.device),
        }

    def __call__(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[feature_dim]`` fp32 features of one encoded image."""

        self._pooled.clear()
        with torch.inference_mode():
            output = self.model(
                encoded["pixel_values"], grid_thw=encoded["image_grid_thw"]
            )
        missing = [i for i in self.settings.taps if i not in self._pooled]
        if missing:
            raise TowerError(f"这些块没有输出: {missing}")
        parts = [self._pooled[i] for i in self.settings.taps]
        parts.append(output.pooler_output.float().mean(dim=0))
        return torch.cat(parts)


def load_tower(
    models_dir: Path, settings: TowerSettings, device: str
) -> tuple[TowerFeatures, str]:
    """Verify and load the extracted tower; returns it and its weights hash."""

    check_runtime(device, settings.precision)
    weights = verify_tower(models_dir)
    tower_dir = models_dir / TOWER_DIR
    dtype = torch.bfloat16 if settings.precision == "bf16" else torch.float32
    model = Qwen3_5VisionModel.from_pretrained(
        tower_dir, dtype=dtype, attn_implementation="sdpa"
    )
    processor = AutoImageProcessor.from_pretrained(tower_dir)
    features = TowerFeatures(model, processor, settings, device)
    return features, weights.find(TOWER_WEIGHTS_FILE).sha256
