from __future__ import annotations

import torch
from torch import nn

from mlx.modes.image_classification.models.base import BaseImageSimilarityModel


class SiameseBackbone(BaseImageSimilarityModel):
    """Shared-weight Siamese model backed by a standard classifier architecture."""

    def __init__(self, backbone: nn.Module, *, embedding_size: int) -> None:
        super().__init__()
        if embedding_size < 1:
            raise ValueError("embedding_size must be at least 1.")

        self.embedding_size = embedding_size
        self.embedding = backbone
        self.embedding_activation = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_activation(self.embedding(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return self.fc(torch.abs(out1 - out2))
