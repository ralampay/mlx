from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import count_model_parameters
from mlx.modes.segmentation.models import (
    SMALL_MODEL_NAMES as SEGMENTATION_SMALL_MODEL_NAMES,
    build_segmentation_model,
    supported_model_names as segmentation_model_names,
)
from mlx.modes.segmentation.models.backbones import BACKBONE_SPECS

DEFAULT_MODEL = "unet"
MODEL_NAMES = frozenset(segmentation_model_names())
SMALL_MODEL_NAMES = frozenset(SEGMENTATION_SMALL_MODEL_NAMES)
MODEL_GROUP_NAMES = frozenset({"all", "all-small"})


@dataclass(frozen=True)
class SaliencyModelSummary:
    model_name: str
    parameter_count: int
    backbone: str
    pretrained_supported: bool
    groups: tuple[str, ...]
    output_channels: int = 1


def supported_model_names() -> list[str]:
    return sorted(MODEL_NAMES)


def grouped_model_names(selector: str) -> list[str]:
    if selector == "all":
        return supported_model_names()
    if selector == "all-small":
        return sorted(SMALL_MODEL_NAMES)
    if selector in MODEL_NAMES:
        return [selector]
    available = ", ".join((*sorted(MODEL_GROUP_NAMES), *supported_model_names()))
    raise MLXUserError(
        f"Unsupported saliency model or group '{selector}'. Available values: {available}."
    )


def build_saliency_model(model_name: str, config: dict[str, Any]):
    if model_name not in MODEL_NAMES:
        grouped_model_names(model_name)
    # The shared U-Net implementation already makes its task head configurable.
    # One output channel gives saliency logits without changing encoder/decoder math.
    return build_segmentation_model(model_name, config, num_classes=1)


def model_summary(model_name: str, config: dict[str, Any]) -> SaliencyModelSummary:
    listing_config = {**config, "pretrained": False}
    model = build_saliency_model(model_name, listing_config)
    spec = BACKBONE_SPECS.get(model_name)
    groups = ["all"]
    if model_name in SMALL_MODEL_NAMES:
        groups.append("all-small")
    backbone = "native" if spec is None else spec.classification_model
    pretrained_supported = spec is not None and spec.classification_model != "draxnet"
    summary = SaliencyModelSummary(
        model_name=model_name,
        parameter_count=count_model_parameters(model),
        backbone=backbone,
        pretrained_supported=pretrained_supported,
        groups=tuple(groups),
    )
    del model
    return summary


__all__ = [
    "DEFAULT_MODEL",
    "MODEL_GROUP_NAMES",
    "MODEL_NAMES",
    "SMALL_MODEL_NAMES",
    "SaliencyModelSummary",
    "build_saliency_model",
    "grouped_model_names",
    "model_summary",
    "supported_model_names",
]
