"""Segmentation construction is selected here, independently of workflow code."""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping
from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference
from mlx.modes.segmentation.models.backbones import BACKBONE_SPECS


def build_unet(name, config, *, num_classes):
    from .unet import UNet
    return UNet(in_channels=3 if config.get("colored", True) else 1, num_classes=num_classes)


def build_backbone_unet(name, config, *, num_classes):
    from .backbones import build_segmentation_encoder
    from .unet import BackboneUNet
    return BackboneUNet(build_segmentation_encoder(name, config), num_classes=num_classes)


@dataclass(frozen=True)
class SegmentationModelRegistry:
    entries: Mapping[str, Callable | str] = field(default_factory=lambda: {
        "unet": build_unet, **{name: build_backbone_unet for name in BACKBONE_SPECS},
    })

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def register(self, name, builder):
        if not name.strip():
            raise ValueError("Segmentation model name cannot be empty.")
        return SegmentationModelRegistry({**self.entries, name.strip().lower(): builder})

    def resolve(self, name):
        value = name if ":" in name else self.entries.get(name)
        if value is None:
            raise MLXUserError(f"Unsupported segmentation model '{name}'. Available models: {', '.join(sorted(self.entries))}.")
        builder = load_reference(value, kind="segmentation model") if isinstance(value, str) else value
        if not callable(builder):
            raise MLXUserError(f"Segmentation model '{name}' builder must be callable.")
        return builder


DEFAULT_SEGMENTATION_REGISTRY = SegmentationModelRegistry()
