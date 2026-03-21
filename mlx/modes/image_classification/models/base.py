from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseImageSimilarityModel(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass comparing two input tensors."""

    def predict(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x1, x2)
