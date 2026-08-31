from __future__ import annotations

from mlx.core.image_backbones import ImageFeatureBackbone


def image_backbone_names() -> tuple[str, ...]:
    """Return classifier-owned standard backbones through one integration gateway."""

    from mlx.modes.image_classification.models import standard_model_names

    return tuple(standard_model_names())


def build_image_backbone(model_name: str, config: dict) -> ImageFeatureBackbone:
    from mlx.modes.image_classification.models.adapters import build_image_feature_backbone

    return build_image_feature_backbone(model_name, config)


def backbone_class_name(backbone: ImageFeatureBackbone) -> str:
    adapter = getattr(backbone, "adapter", None)
    concrete = getattr(adapter, "model", backbone)
    return type(concrete).__name__


__all__ = ["backbone_class_name", "build_image_backbone", "image_backbone_names"]
