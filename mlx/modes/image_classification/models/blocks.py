from __future__ import annotations

import torch
from torch import nn


class ConvActivationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_factory=nn.ReLU,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation_factory(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvActivationPoolBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_factory=nn.ReLU,
        pool_factory=nn.MaxPool2d,
        pool_kernel_size: int = 2,
        pool_stride: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvActivationBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation_factory=activation_factory,
            ),
            pool_factory(pool_kernel_size, stride=pool_stride or pool_kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
