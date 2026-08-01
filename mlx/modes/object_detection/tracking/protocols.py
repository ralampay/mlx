from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from mlx.modes.object_detection.tracking.models import (
    TrackingDetection,
    TrackingFrameResult,
)


@runtime_checkable
class TrackingAlgorithm(Protocol):
    """Structural interface implemented by online tracking algorithms."""

    def update(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        """Process externally computed detections for one frame."""
        ...

    def reset(self) -> None:
        """Discard state belonging to the current tracking session."""
        ...
