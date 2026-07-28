from __future__ import annotations

import torch
from torch import nn

from mlx.modes.segmentation.models.base import BaseSegmentationModel
from mlx.modes.segmentation.models.blocks import (
    DoubleConvBlock,
    DownsampleConvBlock,
    UNetDecoderBlock,
    UpsampleSkipConvBlock,
)
from mlx.modes.segmentation.models.backbones import SegmentationEncoder


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


class BackboneUNet(BaseSegmentationModel):
    def __init__(
        self,
        encoder: SegmentationEncoder,
        *,
        num_classes: int = 2,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        if len(encoder.output_channels) not in (4, 5):
            raise ValueError("U-Net encoders must provide four or five feature levels.")
        if len(decoder_channels) != 5:
            raise ValueError("Backbone U-Net requires five decoder channel widths.")

        self.encoder = encoder
        skip_channels = list(reversed(encoder.output_channels[:-1]))
        skip_channels.extend([0] * (len(decoder_channels) - len(skip_channels)))
        input_channels = encoder.output_channels[-1]
        blocks = []
        for skip_width, output_width in zip(
            skip_channels,
            decoder_channels,
            strict=True,
        ):
            blocks.append(UNetDecoderBlock(input_channels, skip_width, output_width))
            input_channels = output_width
        self.decoder = nn.ModuleList(blocks)
        self.outc = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        if len(features) != len(self.encoder.output_channels):
            raise RuntimeError(
                "Segmentation encoder returned a different number of features than declared."
            )

        x = features[-1]
        skips = list(reversed(features[:-1]))
        for index, decoder in enumerate(self.decoder):
            skip = skips[index] if index < len(skips) else None
            x = decoder(x, skip)
        x = self.outc(x)
        if x.shape[-2:] != input_size:
            x = nn.functional.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return x
