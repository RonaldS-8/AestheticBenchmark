"""Backbone plus two score heads.

The backbone's stage outputs are average-pooled and concatenated (as in the
probe), LayerNorm-ed, and fed to one small MLP per score. Heads predict
z-scored targets; ``to_scores`` maps back to the 1-5 scale with the training
mean and standard deviation stored in the model.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from aes_head.backbones import MultiLevelExtractor, get_spec
from aes_head.schema import EVA_DIMENSIONS


def strip_classifier(model: nn.Module) -> None:
    """Replace the ImageNet classifier with identity; it is never used."""

    for name in ("fc", "classifier"):
        if hasattr(model, name):
            setattr(model, name, nn.Identity())


class HeadModel(nn.Module):
    """Multi-level pooled backbone features to aesthetic and technical scores."""

    def __init__(
        self,
        backbone: str,
        model: nn.Module,
        *,
        hidden_dim: int,
        dropout: float,
        target_mean: Sequence[float],
        target_std: Sequence[float],
    ) -> None:
        super().__init__()
        spec = get_spec(backbone)
        strip_classifier(model)
        self.backbone_name = backbone
        self.extractor = MultiLevelExtractor(spec, model)
        width = sum(spec.dims)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in EVA_DIMENSIONS
        )
        self.register_buffer("target_mean", torch.tensor(target_mean))
        self.register_buffer("target_std", torch.tensor(target_std))

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the pretrained backbone."""

        return self.extractor.model.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters added on top: the LayerNorm and both heads."""

        yield from self.norm.parameters()
        yield from self.heads.parameters()

    def forward(
        self, images: Sequence[torch.Tensor], *, train_backbone: bool = True
    ) -> torch.Tensor:
        """``[batch, 2]`` z-scores; each image ``[3, H, W]`` runs at its own size."""

        with torch.set_grad_enabled(train_backbone and torch.is_grad_enabled()):
            pooled = [torch.cat(self.extractor(image[None]), dim=1) for image in images]
        features = self.norm(torch.cat(pooled).float())
        return torch.cat([head(features) for head in self.heads], dim=1)

    def to_scores(self, z: torch.Tensor) -> torch.Tensor:
        """Map z-scores back to the 1-5 target scale."""

        return z.float() * self.target_std + self.target_mean
