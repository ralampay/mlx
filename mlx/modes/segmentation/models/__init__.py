from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.models.unet import UNet

DEFAULT_MODEL = "unet"
MODEL_NAMES = {"unet"}


def supported_model_names() -> list[str]:
    return sorted(MODEL_NAMES)


def build_segmentation_model(
    model_name: str,
    config: dict[str, Any],
    *,
    num_classes: int,
):
    if model_name != "unet":
        available = ", ".join(supported_model_names())
        raise MLXUserError(
            f"Unsupported segmentation model '{model_name}'. Available models: {available}."
        )

    return UNet(
        in_channels=3 if config.get("colored", True) else 1,
        num_classes=num_classes,
    )


__all__ = ["DEFAULT_MODEL", "MODEL_NAMES", "UNet", "build_segmentation_model", "supported_model_names"]

