from __future__ import annotations

from typing import Any, Protocol

from torch import nn


class ClassificationBackboneFactory(Protocol):
    """Builds a classifier-shaped feature extractor for a segmentation encoder."""

    def __call__(
        self,
        model_name: str,
        config: dict[str, Any],
        *,
        num_classes: int,
    ) -> nn.Module:
        ...


def build_default_classification_backbone(
    model_name: str,
    config: dict[str, Any],
    *,
    num_classes: int,
) -> nn.Module:
    """Compatibility adapter for the existing shared classifier backbones.

    The dependency on image classification is isolated here so the segmentation
    encoder and U-Net implementation depend only on an injected factory.
    """

    from mlx.modes.image_classification.models import build_image_classification_model

    return build_image_classification_model(
        model_name,
        config,
        num_classes=num_classes,
    )


__all__ = [
    "ClassificationBackboneFactory",
    "build_default_classification_backbone",
]
