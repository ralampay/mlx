from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking.models import (
    TrackingDetection,
    TrackingFrameResult,
)
from mlx.modes.object_detection.tracking.protocols import TrackingAlgorithm


class RunTrackByDetectionCommand:
    """Coordinate one frame of detection-driven online tracking."""

    def __init__(self, *, algorithm: TrackingAlgorithm) -> None:
        self._algorithm = algorithm
        self._frame_index = 0

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def execute(
        self,
        *,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        validate_tracking_frame(frame)
        self._frame_index += 1
        return self._algorithm.update(
            frame_index=self._frame_index,
            detections=detections,
            frame=frame,
        )

    def reset(self) -> None:
        self._frame_index = 0
        self._algorithm.reset()


def validate_tracking_frame(frame: np.ndarray | None) -> None:
    """Validate a frame without retaining it."""

    if frame is None:
        return
    if not isinstance(frame, np.ndarray):
        raise MLXUserError("Tracking frame must be a NumPy array when provided.")
    if frame.ndim not in (2, 3):
        raise MLXUserError(
            "Tracking frame must have shape (height, width) or "
            "(height, width, channels)."
        )
    if frame.size == 0 or any(dimension == 0 for dimension in frame.shape):
        raise MLXUserError("Tracking frame must not be empty.")
