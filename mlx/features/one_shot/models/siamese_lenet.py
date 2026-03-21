import torch
import torch.nn as nn

from mlx.features.one_shot.models.base import BaseICOneShotModel


class SiameseLeNet(BaseICOneShotModel):
    def __init__(self, colored: bool = True, embedding_size: int = 4096) -> None:
        super().__init__()

        channels = 3 if colored else 1
        self.embedding_size = embedding_size

        self.embedding = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, embedding_size),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = torch.abs(out1 - out2)
        return self.fc(diff)
