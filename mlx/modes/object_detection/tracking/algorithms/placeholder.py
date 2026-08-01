from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.tracking.models import (
    TrackResult,
    TrackStatus,
    TrackingDetection,
    TrackingFrameResult,
)


class DetectionAsTrackAlgorithm:
    """Skeletal algorithm that emits every detection as a new confirmed track.

    This implementation performs no temporal association. It retains only the next
    identifier, making it useful for exercising the generic API before a real online
    tracking algorithm is selected.
    """

    def __init__(self) -> None:
        self._next_track_id = 1

    def update(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        if frame_index < 0:
            raise ValueError("Tracking frame index must be zero or greater.")
        tracks = tuple(
            self._new_track(detection=detection, frame_index=frame_index)
            for detection in detections
        )
        return TrackingFrameResult(frame_index=frame_index, tracks=tracks)

    def reset(self) -> None:
        self._next_track_id = 1

    def _new_track(
        self,
        *,
        detection: TrackingDetection,
        frame_index: int,
    ) -> TrackResult:
        track_id = self._next_track_id
        self._next_track_id += 1
        return TrackResult(
            track_id=track_id,
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            status=TrackStatus.CONFIRMED,
            hits=1,
            missing_frames=0,
            last_seen_frame=frame_index,
        )
