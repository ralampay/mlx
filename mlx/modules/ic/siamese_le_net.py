import torch
import torch.nn as nn
from .base_ic_one_shot_model import BaseICOneShotModel

class SiameseLeNet(BaseICOneShotModel):
    def __init__(self, colored=True, embedding_size=4096):
        super().__init__()

        self.colored = colored
        self.input_size = 3 if colored else 1
        self.embedding_size = 4096

        self.embedding = nn.Sequential(
            nn.Conv2d(self.input_size, 64, kernel_size=10),
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
            nn.Flatten(),

            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for one input image."""
        return self.embedding(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass for a pair of images."""
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = torch.abs(out1 - out2)

        return self.fc(diff)
