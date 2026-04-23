from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models.blocks import DraxBlock


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity
        return self.relu(x)


class DraxResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_attention: bool = True,
        efficient_attention: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drax = DraxBlock(
            dim=out_channels,
            use_attention=use_attention,
            efficient=efficient_attention,
        )
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.drax(x)
        x = self.proj(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity
        return self.relu(x)


class DraxNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        block: type[BasicResidualBlock] = BasicResidualBlock,
        layers: Sequence[int] = (2, 2, 2, 2),
        stage_block_types: Sequence[str] = ("basic", "basic", "basic", "drax"),
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        if tuple(layers) != (2, 2, 2, 2):
            raise ValueError("DraxNet currently implements ResNet-18 only, so layers must be (2, 2, 2, 2).")
        if len(stage_block_types) != len(layers):
            raise ValueError("stage_block_types must have one entry per ResNet stage.")

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_blocks = [self._resolve_stage_block(block_type, default_block=block) for block_type in stage_block_types]
        self.layer1 = self._make_layer(stage_blocks[0], 64, layers[0])
        self.layer2 = self._make_layer(stage_blocks[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(stage_blocks[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(stage_blocks[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicResidualBlock):
                    nn.init.constant_(module.bn2.weight, 0)
                elif isinstance(module, DraxResidualBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _resolve_stage_block(self, block_type: str, *, default_block: type[BasicResidualBlock]) -> type[nn.Module]:
        normalized = block_type.strip().lower()
        if normalized == "basic":
            return default_block
        if normalized in {"cax", "drax"}:
            return DraxResidualBlock
        raise ValueError(f"Unsupported DraxNet stage block type '{block_type}'.")

    def _make_layer(
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        *,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list[nn.Module] = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_draxnet(*, num_classes: int, colored: bool, pretrained: bool, config: dict | None = None):
    config = config or {}
    model = DraxNet(
        in_channels=3 if colored else 1,
        num_classes=num_classes,
        stage_block_types=tuple(config.get("draxnet_stage_blocks", "basic,basic,basic,drax").split(",")),
    )

    if pretrained:
        if tuple(config.get("draxnet_stage_blocks", "basic,basic,basic,drax").split(",")) != ("basic", "basic", "basic", "basic"):
            raise MLXUserError("Pretrained DraxNet weights currently require all stages to use basic blocks.")
        try:
            from torchvision import models as torchvision_models
        except ImportError as exc:
            raise MLXUserError(
                "Torchvision is required for pretrained DraxNet weights. Install it with 'pip install torchvision'."
            ) from exc

        reference = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.DEFAULT)
        state_dict = reference.state_dict()
        if not colored:
            state_dict["conv1.weight"] = state_dict["conv1.weight"].mean(dim=1, keepdim=True)
        state_dict["fc.weight"] = model.fc.weight.detach()
        state_dict["fc.bias"] = model.fc.bias.detach()
        model.load_state_dict(state_dict)

    return model
