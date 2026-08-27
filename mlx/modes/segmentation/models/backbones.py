from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.models.backbone_factory import (
    ClassificationBackboneFactory,
    build_default_classification_backbone,
)


@dataclass(frozen=True)
class BackboneSpec:
    classification_model: str
    encoder_family: str
    output_channels: tuple[int, ...]
    fusion_mode: str | None = None


BACKBONE_SPECS = {
    "unet-resnet18": BackboneSpec(
        "resnet18",
        "residual",
        (64, 64, 128, 256, 512),
    ),
    "unet-resnet50": BackboneSpec(
        "resnet50",
        "residual",
        (64, 256, 512, 1024, 2048),
    ),
    "unet-densenet121": BackboneSpec(
        "densenet121",
        "densenet",
        (64, 256, 512, 1024, 1024),
    ),
    "unet-mobilenet_v3_large": BackboneSpec(
        "mobilenet_v3_large",
        "mobilenet",
        (16, 24, 40, 112, 960),
    ),
    "unet-efficientnet_b0": BackboneSpec(
        "efficientnet_b0",
        "efficientnet",
        (32, 24, 40, 112, 1280),
    ),
    "unet-convnext_tiny": BackboneSpec(
        "convnext_tiny",
        "convnext",
        (96, 192, 384, 768),
    ),
    "unet-convnext_small": BackboneSpec(
        "convnext_small",
        "convnext",
        (96, 192, 384, 768),
    ),
    "unet-convnext_base": BackboneSpec(
        "convnext_base",
        "convnext",
        (128, 256, 512, 1024),
    ),
    "unet-convnext_large": BackboneSpec(
        "convnext_large",
        "convnext",
        (192, 384, 768, 1536),
    ),
    "unet-draxnet-average": BackboneSpec(
        "draxnet",
        "residual",
        (64, 64, 128, 256, 512),
        "average",
    ),
    "unet-draxnet-sknet": BackboneSpec(
        "draxnet",
        "residual",
        (64, 64, 128, 256, 512),
        "sknet",
    ),
    "unet-drax_mobilenet_v3_large-average": BackboneSpec(
        "drax_mobilenet_v3_large",
        "drax_mobilenet",
        (16, 24, 40, 112, 960),
        "average",
    ),
    "unet-drax_mobilenet_v3_large-sknet": BackboneSpec(
        "drax_mobilenet_v3_large",
        "drax_mobilenet",
        (16, 24, 40, 112, 960),
        "sknet",
    ),
}


class SegmentationEncoder(nn.Module):
    output_channels: tuple[int, ...]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class ResidualEncoder(SegmentationEncoder):
    def __init__(self, model: nn.Module, output_channels: tuple[int, ...]) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.maxpool = model.maxpool
        self.stages = nn.ModuleList(
            [model.layer1, model.layer2, model.layer3, model.layer4]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)]
        x = self.maxpool(features[0])
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class DenseNetEncoder(SegmentationEncoder):
    def __init__(self, model: nn.Module, output_channels: tuple[int, ...]) -> None:
        super().__init__()
        self.output_channels = output_channels
        features = model.features
        self.stem = nn.Sequential(features.conv0, features.norm0, features.relu0)
        self.pool0 = features.pool0
        self.blocks = nn.ModuleList(
            [
                features.denseblock1,
                features.denseblock2,
                features.denseblock3,
                features.denseblock4,
            ]
        )
        self.transitions = nn.ModuleList(
            [features.transition1, features.transition2, features.transition3]
        )
        self.final_norm = features.norm5

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stem = self.stem(x)
        x = self.pool0(stem)
        dense1 = self.blocks[0](x)
        dense2 = self.blocks[1](self.transitions[0](dense1))
        dense3 = self.blocks[2](self.transitions[1](dense2))
        dense4 = self.blocks[3](self.transitions[2](dense3))
        return [
            stem,
            dense1,
            dense2,
            dense3,
            nn.functional.relu(self.final_norm(dense4), inplace=True),
        ]


class SequentialStageEncoder(SegmentationEncoder):
    def __init__(
        self,
        features: nn.Sequential,
        *,
        stage_ends: tuple[int, ...],
        output_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        children = list(features.children())
        stage_starts = (0, *stage_ends[:-1])
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*children[start:end])
                for start, end in zip(stage_starts, stage_ends, strict=True)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class DraxMobileNetEncoder(SequentialStageEncoder):
    def __init__(
        self,
        model: nn.Module,
        output_channels: tuple[int, ...],
    ) -> None:
        super().__init__(
            model.features,
            stage_ends=(1, 4, 7, 13, 17),
            output_channels=output_channels,
        )
        self.adapter_down = model.adapter_down
        self.adapter_norm = model.adapter_norm
        self.adapter_activation = model.adapter_activation
        self.drax_refiner = model.drax_refiner
        self.adapter_up = model.adapter_up
        self.adapter_up_norm = model.adapter_up_norm

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = super().forward(x)
        residual = outputs[-1]
        refined = self.adapter_down(residual)
        refined = self.adapter_norm(refined)
        refined = self.adapter_activation(refined)
        refined = self.drax_refiner(refined)
        refined = self.adapter_up(refined)
        refined = self.adapter_up_norm(refined)
        outputs[-1] = residual + refined
        return outputs


def build_segmentation_encoder(
    model_name: str,
    config: dict[str, Any],
    *,
    backbone_factory: ClassificationBackboneFactory | None = None,
) -> SegmentationEncoder:
    try:
        spec = BACKBONE_SPECS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(BACKBONE_SPECS))
        raise MLXUserError(
            f"Unsupported U-Net backbone model '{model_name}'. Available backbones: {available}."
        ) from exc

    model_config = dict(config)
    if spec.classification_model == "draxnet" and spec.fusion_mode is not None:
        model_config["draxnet_fusion_mode"] = spec.fusion_mode
    if (
        spec.classification_model == "drax_mobilenet_v3_large"
        and spec.fusion_mode is not None
    ):
        model_config["drax_mobilenet_fusion_mode"] = spec.fusion_mode

    factory = backbone_factory or build_default_classification_backbone
    classifier = factory(
        spec.classification_model,
        model_config,
        num_classes=1,
    )
    if spec.encoder_family == "residual":
        return ResidualEncoder(classifier, spec.output_channels)
    if spec.encoder_family == "densenet":
        return DenseNetEncoder(classifier, spec.output_channels)
    if spec.encoder_family == "mobilenet":
        return SequentialStageEncoder(
            classifier.features,
            stage_ends=(1, 4, 7, 13, 17),
            output_channels=spec.output_channels,
        )
    if spec.encoder_family == "efficientnet":
        return SequentialStageEncoder(
            classifier.features,
            stage_ends=(1, 3, 4, 6, 9),
            output_channels=spec.output_channels,
        )
    if spec.encoder_family == "convnext":
        return SequentialStageEncoder(
            classifier.features,
            stage_ends=(2, 4, 6, 8),
            output_channels=spec.output_channels,
        )
    if spec.encoder_family == "drax_mobilenet":
        return DraxMobileNetEncoder(classifier, spec.output_channels)
    raise MLXUserError(
        f"U-Net backbone '{model_name}' has unsupported encoder family '{spec.encoder_family}'."
    )


__all__ = [
    "BACKBONE_SPECS",
    "BackboneSpec",
    "SegmentationEncoder",
    "build_segmentation_encoder",
]
