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


class ByteTrackAlgorithm:
    """Compact ByteTrack implementation with high/low-confidence association."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        new_track_threshold: float = 0.7,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
    ) -> None:
        validate_common_settings(
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )
        for name, value in (
            ("low_threshold", low_threshold),
            ("high_threshold", high_threshold),
            ("new_track_threshold", new_track_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if low_threshold > high_threshold:
            raise ValueError("low_threshold cannot be greater than high_threshold.")
        if new_track_threshold < high_threshold:
            raise ValueError(
                "new_track_threshold must be greater than or equal to high_threshold."
            )
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.new_track_threshold = float(new_track_threshold)
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
        for track in self._tracks:
            track.predict()

        high_detections = tuple(
            detection
            for detection in detections
            if detection.confidence >= self.high_threshold
        )
        low_detections = tuple(
            detection
            for detection in detections
            if self.low_threshold <= detection.confidence < self.high_threshold
        )
        high_matches, unmatched_track_indices, unmatched_high_indices = (
            associate_detections(
                self._tracks,
                high_detections,
                iou_threshold=self.iou_threshold,
            )
        )
        for track_index, detection_index in high_matches:
            self._update_track(
                track_index,
                high_detections[detection_index],
                frame_index=frame_index,
            )

        remaining_tracks = [self._tracks[index] for index in unmatched_track_indices]
        low_matches, _, _ = associate_detections(
            remaining_tracks,
            low_detections,
            iou_threshold=self.iou_threshold,
        )
        for remaining_index, detection_index in low_matches:
            original_index = unmatched_track_indices[remaining_index]
            self._update_track(
                original_index,
                low_detections[detection_index],
                frame_index=frame_index,
            )

        for detection_index in unmatched_high_indices:
            detection = high_detections[detection_index]
            if detection.confidence >= self.new_track_threshold:
                self._create_track(detection, frame_index=frame_index)

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

    def _update_track(
        self,
        track_index: int,
        detection: TrackingDetection,
        *,
        frame_index: int,
    ) -> None:
        self._tracks[track_index].update(
            detection,
            frame_index=frame_index,
            min_hits=self.min_hits,
        )

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
