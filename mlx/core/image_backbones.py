from __future__ import annotations

from typing import Any, Protocol

import torch


class ImageFeatureBackbone(Protocol):
    """Neutral contract for RGB image models that return penultimate features."""

    @property
    def feature_dim(self) -> int: ...

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image with shape ``[B, D]``."""


class ImageFeatureBackboneFactory(Protocol):
    def __call__(
        self,
        model_name: str,
        config: dict[str, Any],
    ) -> ImageFeatureBackbone:
        ...


__all__ = ["ImageFeatureBackbone", "ImageFeatureBackboneFactory"]
