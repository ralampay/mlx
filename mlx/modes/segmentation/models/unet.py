from __future__ import annotations

import torch
from torch import nn

from mlx.modes.segmentation.models.base import BaseSegmentationModel
from mlx.modes.segmentation.models.blocks import (
    DoubleConvBlock,
    DownsampleConvBlock,
    UpsampleSkipConvBlock,
)


class UNet(BaseSegmentationModel):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.inc = DoubleConvBlock(in_channels, features[0])
        self.down1 = DownsampleConvBlock(features[0], features[1])
        self.down2 = DownsampleConvBlock(features[1], features[2])
        self.down3 = DownsampleConvBlock(features[2], features[3])
        self.bottleneck = DownsampleConvBlock(features[3], features[3] * 2)
        self.up1 = UpsampleSkipConvBlock(features[3] * 2, features[3], features[3])
        self.up2 = UpsampleSkipConvBlock(features[3], features[2], features[2])
        self.up3 = UpsampleSkipConvBlock(features[2], features[1], features[1])
        self.up4 = UpsampleSkipConvBlock(features[1], features[0], features[0])
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
