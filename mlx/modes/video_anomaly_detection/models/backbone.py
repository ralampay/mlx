from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mlx.core.image_backbones import ImageFeatureBackboneFactory


def build_default_frame_backbone(model_name: str, config: dict[str, Any]):
    """Compatibility boundary for the image-classification model registry."""

    from mlx.modes.image_classification.models import build_image_feature_backbone

    return build_image_feature_backbone(model_name, config)


class FrameBackbone(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(backbone.feature_dim)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError("FrameBackbone expects clips shaped [B, T, C, H, W].")
        batch, frames, channels, height, width = clips.shape
        features = self.backbone(
            clips.reshape(batch * frames, channels, height, width)
        )
        if features.ndim != 2 or features.shape[0] != batch * frames:
            raise ValueError("Image feature backbone must return [B*T, D] vectors.")
        return features.reshape(batch, frames, features.shape[-1])


__all__ = [
    "FrameBackbone",
    "ImageFeatureBackboneFactory",
    "build_default_frame_backbone",
]
