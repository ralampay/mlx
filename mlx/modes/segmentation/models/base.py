from __future__ import annotations

from torch import nn


class BaseSegmentationModel(nn.Module):
    """Minimal base class for segmentation models."""

