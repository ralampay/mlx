from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models.base import BaseImageSimilarityModel
from mlx.modes.image_classification.models.siamese_lenet import SiameseLeNet
from mlx.modes.image_classification.models.standard import (
    build_standard_model,
    registered_standard_model_names,
    register_standard_model,
)

DEFAULT_MODEL = "resnet18"
ONE_SHOT_MODEL_NAMES = {"siamese-le-net"}
STANDARD_MODEL_NAMES = {"resnet18", "resnet50"}


def supported_model_names() -> list[str]:
    return sorted(ONE_SHOT_MODEL_NAMES | STANDARD_MODEL_NAMES | set(registered_standard_model_names()))


def model_family_for(model_name: str) -> str:
    if model_name in ONE_SHOT_MODEL_NAMES:
        return "one-shot"
    if model_name in STANDARD_MODEL_NAMES or model_name in registered_standard_model_names():
        return "standard"
    available = ", ".join(supported_model_names())
    raise MLXUserError(f"Unsupported image-classification model '{model_name}'. Available models: {available}.")


def build_image_classification_model(
    model_name: str,
    config: dict[str, Any],
    *,
    num_classes: int | None = None,
):
    family = model_family_for(model_name)
    if family == "one-shot":
        return SiameseLeNet(
            colored=config.get("colored", True),
            embedding_size=config.get("embedding_size", 4096),
        )

    if num_classes is None:
        raise MLXUserError(
            f"Model '{model_name}' requires the number of classes before it can be constructed."
        )

    return build_standard_model(
        model_name,
        num_classes=num_classes,
        colored=config.get("colored", True),
        pretrained=bool(config.get("pretrained", False)),
    )


__all__ = [
    "BaseImageSimilarityModel",
    "DEFAULT_MODEL",
    "ONE_SHOT_MODEL_NAMES",
    "STANDARD_MODEL_NAMES",
    "SiameseLeNet",
    "build_image_classification_model",
    "model_family_for",
    "register_standard_model",
    "supported_model_names",
]
