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

from mlx.modes.segmentation.models.registry import SegmentationModelRegistry, DEFAULT_SEGMENTATION_REGISTRY

DEFAULT_MODEL = "unet"
MODEL_NAMES = frozenset(DEFAULT_SEGMENTATION_REGISTRY.entries)
SMALL_MODEL_NAMES = frozenset(
    {
        "unet-mobilenet_v3_large",
        "unet-drax_mobilenet_v3_large-average",
        "unet-drax_mobilenet_v3_large-sknet",
        "unet-efficientnet_b0",
    }
)
MODEL_GROUP_NAMES = frozenset({"all", "all-small"})


def supported_model_names(registry=None) -> list[str]:
    return sorted((registry or DEFAULT_SEGMENTATION_REGISTRY).entries)


def grouped_model_names(group_name: str, *, registry=None) -> list[str]:
    if group_name == "all":
        return supported_model_names(registry)
    if group_name == "all-small":
        return sorted(SMALL_MODEL_NAMES & set(supported_model_names(registry)))
    available = ", ".join(sorted(MODEL_GROUP_NAMES))
    raise MLXUserError(f"Unsupported segmentation model group '{group_name}': {available}.")


def build_segmentation_model(
    model_name: str,
    config: dict[str, Any],
    *,
    num_classes: int,
    registry: SegmentationModelRegistry | None = None,
):
    builder = (registry or DEFAULT_SEGMENTATION_REGISTRY).resolve(model_name)
    try:
        return builder(model_name, config, num_classes=num_classes)
    except (ValueError, TypeError) as exc:
        raise MLXUserError(f"Cannot build segmentation model '{model_name}': {exc}") from exc


__all__ = [
    "ConvNormAct",
    "DEFAULT_MODEL",
    "BackboneUNet",
    "DoubleConvBlock",
    "DownsampleConvBlock",
    "MODEL_NAMES",
    "MODEL_GROUP_NAMES",
    "SMALL_MODEL_NAMES",
    "SegmentationEncoder",
    "UNet",
    "UNetDecoderBlock",
    "UpsampleSkipConvBlock",
    "build_segmentation_encoder",
    "build_segmentation_model",
    "grouped_model_names",
    "supported_model_names",
]
