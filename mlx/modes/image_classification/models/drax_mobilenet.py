from __future__ import annotations

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models.blocks import DraxBlock


class DraxMobileNetV3Large(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,
        drax_blocks: int = 1,
        adapter_dim: int = 160,
        use_attention: bool = True,
        efficient_attention: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if drax_blocks < 1:
            raise ValueError("drax_blocks must be at least 1.")

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

        final_channels = backbone.classifier[0].in_features
        if adapter_dim < 1:
            raise ValueError("adapter_dim must be at least 1.")

        self.adapter_down = nn.Conv2d(final_channels, adapter_dim, kernel_size=1, bias=False)
        self.adapter_norm = nn.BatchNorm2d(adapter_dim)
        self.adapter_activation = nn.Hardswish()
        self.drax_refiner = nn.Sequential(
            *[
                DraxBlock(
                    dim=adapter_dim,
                    use_attention=use_attention,
                    efficient=efficient_attention,
                    drop_path=drop_path,
                )
                for _ in range(drax_blocks)
            ]
        )
        self.adapter_up = nn.Conv2d(adapter_dim, final_channels, kernel_size=1, bias=False)
        self.adapter_up_norm = nn.BatchNorm2d(final_channels)
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        residual = x
        x = self.adapter_down(x)
        x = self.adapter_norm(x)
        x = self.adapter_activation(x)
        x = self.drax_refiner(x)
        x = self.adapter_up(x)
        x = self.adapter_up_norm(x)
        x = residual + x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _replace_mobilenet_stem_conv(model: nn.Module) -> None:
    original_conv = model.features[0][0]
    replacement_conv = nn.Conv2d(
        1,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    with torch.no_grad():
        replacement_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        if original_conv.bias is not None and replacement_conv.bias is not None:
            replacement_conv.bias.copy_(original_conv.bias)
    model.features[0][0] = replacement_conv


def build_drax_mobilenet_v3_large(*, num_classes: int, colored: bool, pretrained: bool, config: dict | None = None):
    config = config or {}
    try:
        from torchvision import models as torchvision_models
    except ImportError as exc:
        raise MLXUserError(
            "Torchvision is required for Drax MobileNet V3 Large. Install it with 'pip install torchvision'."
        ) from exc

    weights = torchvision_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    backbone = torchvision_models.mobilenet_v3_large(weights=weights)

    if not colored:
        _replace_mobilenet_stem_conv(backbone)

    return DraxMobileNetV3Large(
        backbone,
        num_classes=num_classes,
        drax_blocks=int(config.get("drax_mobilenet_blocks", 1)),
        adapter_dim=int(config.get("drax_mobilenet_adapter_dim", 160)),
        use_attention=bool(config.get("drax_mobilenet_use_attention", True)),
        efficient_attention=bool(config.get("drax_mobilenet_efficient_attention", True)),
        drop_path=float(config.get("drax_mobilenet_drop_path", 0.0)),
    )
