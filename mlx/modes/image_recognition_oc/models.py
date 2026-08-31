from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mlx.core.deep_svdd import squared_euclidean_score
from mlx.core.exceptions import MLXUserError
from mlx.core.image_backbones import ImageFeatureBackbone


@dataclass(frozen=True)
class ImageOneClassOutput:
    embedding: torch.Tensor
    anomaly_score: torch.Tensor


class DeepSVDDImageRecognizer(nn.Module):
    def __init__(
        self,
        backbone: ImageFeatureBackbone,
        *,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if min(int(backbone.feature_dim), hidden_dim, embedding_dim) < 1:
            raise MLXUserError("Deep SVDD dimensions must be positive.")
        self.backbone = backbone
        self.backbone_feature_dim = int(backbone.feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_feature_dim, hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(embedding_dim))
        self.register_buffer("threshold", torch.tensor(float("nan")))

    def forward(self, images: torch.Tensor) -> ImageOneClassOutput:
        embedding = self.projection(self.backbone(images))
        return ImageOneClassOutput(
            embedding=embedding,
            anomaly_score=squared_euclidean_score(embedding, self.center),
        )

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(self.threshold):
            raise MLXUserError(
                "The one-class checkpoint has no calibrated threshold. "
                "Use a completed deployment or resumable checkpoint."
            )
        return self(images).anomaly_score > self.threshold


__all__ = ["DeepSVDDImageRecognizer", "ImageOneClassOutput"]
