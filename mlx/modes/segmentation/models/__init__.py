from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.models.blocks import (
    ConvNormAct,
    DoubleConvBlock,
    DownsampleConvBlock,
    UNetDecoderBlock,
    UpsampleSkipConvBlock,
)
from mlx.modes.segmentation.models.backbones import (
    BACKBONE_SPECS,
    SegmentationEncoder,
    build_segmentation_encoder,
)
from mlx.modes.segmentation.models.unet import BackboneUNet, UNet

DEFAULT_MODEL = "unet"
MODEL_NAMES = {"unet", *BACKBONE_SPECS}


def supported_model_names() -> list[str]:
    return sorted(MODEL_NAMES)


def build_segmentation_model(
    model_name: str,
    config: dict[str, Any],
    *,
    num_classes: int,
):
    if model_name not in MODEL_NAMES:
        available = ", ".join(supported_model_names())
        raise MLXUserError(
            f"Unsupported segmentation model '{model_name}'. Available models: {available}."
        )

    if model_name == "unet":
        return UNet(
            in_channels=3 if config.get("colored", True) else 1,
            num_classes=num_classes,
        )

    try:
        encoder = build_segmentation_encoder(model_name, config)
        return BackboneUNet(encoder, num_classes=num_classes)
    except ValueError as exc:
        raise MLXUserError(f"Cannot build segmentation model '{model_name}': {exc}") from exc


__all__ = [
    "ConvNormAct",
    "DEFAULT_MODEL",
    "BackboneUNet",
    "DoubleConvBlock",
    "DownsampleConvBlock",
    "MODEL_NAMES",
    "SegmentationEncoder",
    "UNet",
    "UNetDecoderBlock",
    "UpsampleSkipConvBlock",
    "build_segmentation_encoder",
    "build_segmentation_model",
    "supported_model_names",
]
