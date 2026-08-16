from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.tracking.algorithms.motion import (
    MotionTrack,
    associate_detections,
    validate_common_settings,
)
from mlx.modes.object_detection.tracking.models import (
    TrackingDetection,
    TrackingFrameResult,
)


class SortTrackingAlgorithm:
    """Compact class-aware SORT implementation using Kalman motion and IoU."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        validate_common_settings(
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: list[MotionTrack] = []
        self._next_track_id = 1

    def update(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        del frame
        if frame_index < 0:
            raise ValueError("Tracking frame index must be zero or greater.")
        self._predict_tracks()
        matches, _, unmatched_detection_indices = associate_detections(
            self._tracks,
            detections,
            iou_threshold=self.iou_threshold,
        )
        for track_index, detection_index in matches:
            self._tracks[track_index].update(
                detections[detection_index],
                frame_index=frame_index,
                min_hits=self.min_hits,
            )
        for detection_index in unmatched_detection_indices:
            self._create_track(detections[detection_index], frame_index=frame_index)
        self._tracks = [
            track for track in self._tracks if track.missing_frames <= self.max_age
        ]
        return TrackingFrameResult(
            frame_index=frame_index,
            tracks=tuple(track.snapshot() for track in self._tracks),
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def _predict_tracks(self) -> None:
        for track in self._tracks:
            track.predict()

    def _create_track(
        self,
        detection: TrackingDetection,
        *,
        frame_index: int,
    ) -> None:
        self._tracks.append(
            MotionTrack.create(
                track_id=self._next_track_id,
                detection=detection,
                frame_index=frame_index,
                min_hits=self.min_hits,
            )
        )
        self._next_track_id += 1
