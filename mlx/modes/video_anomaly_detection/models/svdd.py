from __future__ import annotations

import torch
from torch import nn

from mlx.core.deep_svdd import squared_euclidean_score
from mlx.core.exceptions import MLXUserError


class DeepSVDDHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, embedding_dim) < 1:
            raise MLXUserError("Deep SVDD dimensions must be positive.")
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(embedding_dim))
        self.register_buffer("threshold", torch.tensor(float("nan")))

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(clip_embedding)

    def score(self, embedding: torch.Tensor) -> torch.Tensor:
        return squared_euclidean_score(embedding, self.center)

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(self.threshold):
            raise MLXUserError(
                "The video-anomaly checkpoint has no calibrated Deep SVDD threshold."
            )
        return self.score(embedding) > self.threshold


__all__ = ["DeepSVDDHead"]
