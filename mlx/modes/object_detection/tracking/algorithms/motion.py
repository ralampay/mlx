from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from mlx.modes.object_detection.tracking.models import (
    BoundingBox,
    TrackResult,
    TrackStatus,
    TrackingDetection,
)


class BoundingBoxKalmanFilter:
    """Small constant-velocity Kalman filter for ``cx, cy, width, height`` boxes."""

    def __init__(self, bounding_box: BoundingBox) -> None:
        self._state = np.zeros((8, 1), dtype=np.float64)
        self._state[:4, 0] = _box_to_measurement(bounding_box)
        self._transition = np.eye(8, dtype=np.float64)
        self._transition[:4, 4:] = np.eye(4, dtype=np.float64)
        self._observation = np.zeros((4, 8), dtype=np.float64)
        self._observation[:4, :4] = np.eye(4, dtype=np.float64)
        self._covariance = np.eye(8, dtype=np.float64) * 10.0
        self._covariance[4:, 4:] *= 100.0
        self._process_noise = np.eye(8, dtype=np.float64)
        self._process_noise[4:, 4:] *= 0.01
        self._measurement_noise = np.eye(4, dtype=np.float64)

    def predict(self) -> BoundingBox:
        self._state = self._transition @ self._state
        self._covariance = (
            self._transition @ self._covariance @ self._transition.T
            + self._process_noise
        )
        self._state[2:4, 0] = np.maximum(self._state[2:4, 0], 0.0)
        return _measurement_to_box(self._state[:4, 0])

    def update(self, bounding_box: BoundingBox) -> BoundingBox:
        measurement = _box_to_measurement(bounding_box).reshape(4, 1)
        innovation = measurement - self._observation @ self._state
        innovation_covariance = (
            self._observation @ self._covariance @ self._observation.T
            + self._measurement_noise
        )
        gain = (
            self._covariance
            @ self._observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self._state = self._state + gain @ innovation
        self._covariance = (
            np.eye(8, dtype=np.float64) - gain @ self._observation
        ) @ self._covariance
        self._state[2:4, 0] = np.maximum(self._state[2:4, 0], 0.0)
        return _measurement_to_box(self._state[:4, 0])


@dataclass(slots=True)
class MotionTrack:
    track_id: int
    filter: BoundingBoxKalmanFilter
    bounding_box: BoundingBox
    confidence: float
    class_id: int
    label: str | None
    created_at_frame: int
    last_seen_frame: int
    hits: int = 1
    consecutive_hits: int = 1
    missing_frames: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE

    @classmethod
    def create(
        cls,
        *,
        track_id: int,
        detection: TrackingDetection,
        frame_index: int,
        min_hits: int,
    ) -> MotionTrack:
        return cls(
            track_id=track_id,
            filter=BoundingBoxKalmanFilter(detection.bounding_box),
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            created_at_frame=frame_index,
            last_seen_frame=frame_index,
            status=(
                TrackStatus.CONFIRMED if min_hits <= 1 else TrackStatus.TENTATIVE
            ),
        )

    def predict(self) -> None:
        if self.missing_frames > 0:
            self.consecutive_hits = 0
        self.bounding_box = self.filter.predict()
        self.missing_frames += 1
        if self.status is TrackStatus.CONFIRMED:
            self.status = TrackStatus.LOST

    def update(
        self,
        detection: TrackingDetection,
        *,
        frame_index: int,
        min_hits: int,
    ) -> None:
        was_confirmed = self.status is TrackStatus.LOST
        self.bounding_box = self.filter.update(detection.bounding_box)
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.last_seen_frame = frame_index
        self.hits += 1
        self.consecutive_hits += 1
        self.missing_frames = 0
        if was_confirmed or self.consecutive_hits >= min_hits:
            self.status = TrackStatus.CONFIRMED

    def snapshot(self) -> TrackResult:
        return TrackResult(
            track_id=self.track_id,
            bounding_box=self.bounding_box,
            confidence=self.confidence,
            class_id=self.class_id,
            label=self.label,
            status=self.status,
            hits=self.hits,
            missing_frames=self.missing_frames,
            last_seen_frame=self.last_seen_frame,
        )


def associate_detections(
    tracks: Sequence[MotionTrack],
    detections: Sequence[TrackingDetection],
    *,
    iou_threshold: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """Associate tracks and detections by class-aware IoU."""

    if not tracks or not detections:
        return (), tuple(range(len(tracks))), tuple(range(len(detections)))

    scores = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            if track.class_id == detection.class_id:
                scores[track_index, detection_index] = box_iou(
                    track.bounding_box,
                    detection.bounding_box,
                )

    row_indices, column_indices = linear_sum_assignment(1.0 - scores)
    matches = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for track_index, detection_index in zip(row_indices, column_indices):
        if tracks[track_index].class_id != detections[detection_index].class_id:
            continue
        if scores[track_index, detection_index] < iou_threshold:
            continue
        matches.append((int(track_index), int(detection_index)))
        matched_tracks.add(int(track_index))
        matched_detections.add(int(detection_index))

    return (
        tuple(matches),
        tuple(index for index in range(len(tracks)) if index not in matched_tracks),
        tuple(
            index for index in range(len(detections)) if index not in matched_detections
        ),
    )


def validate_common_settings(
    *,
    iou_threshold: float,
    max_age: int,
    min_hits: int,
) -> None:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1.")
    if max_age < 0:
        raise ValueError("max_age must be zero or greater.")
    if min_hits < 1:
        raise ValueError("min_hits must be one or greater.")


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_to_measurement(bounding_box: BoundingBox) -> np.ndarray:
    return np.asarray(
        [
            (bounding_box.x1 + bounding_box.x2) / 2.0,
            (bounding_box.y1 + bounding_box.y2) / 2.0,
            bounding_box.width,
            bounding_box.height,
        ],
        dtype=np.float64,
    )


def _measurement_to_box(measurement: np.ndarray) -> BoundingBox:
    center_x, center_y, width, height = (float(value) for value in measurement)
    width = max(width, 0.0)
    height = max(height, 0.0)
    return BoundingBox(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )
