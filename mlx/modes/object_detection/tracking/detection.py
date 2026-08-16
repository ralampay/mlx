from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.models import Detection, DetectionAdapter
from mlx.modes.object_detection.tracking.command import (
    RunTrackByDetectionCommand,
    validate_tracking_frame,
)
from mlx.modes.object_detection.tracking.models import (
    BoundingBox,
    TrackingDetection,
    TrackingFrameResult,
)
from mlx.modes.object_detection.tracking.protocols import TrackingAlgorithm


def to_tracking_detections(
    detections: Sequence[Detection],
    *,
    class_ids: frozenset[int] | None = None,
) -> tuple[TrackingDetection, ...]:
    return tuple(
        TrackingDetection(
            bounding_box=BoundingBox(*detection.xyxy),
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
        )
        for detection in detections
        if class_ids is None or detection.class_id in class_ids
    )


class RunObjectDetectionTrackingCommand:
    """Compose any normalized detector with any tracking algorithm."""

    def __init__(
        self,
        *,
        detection_model: DetectionAdapter,
        algorithm: TrackingAlgorithm,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self._detection_model = detection_model
        self._tracking_command = RunTrackByDetectionCommand(algorithm=algorithm)
        self._class_ids = frozenset(class_ids) if class_ids else None

    @property
    def frame_index(self) -> int:
        return self._tracking_command.frame_index

    def execute(self, *, frame: np.ndarray) -> TrackingFrameResult:
        validate_tracking_frame(frame)
        detection_result = self._detection_model.predict(frame)
        return self._tracking_command.execute(
            detections=to_tracking_detections(
                detection_result.detections,
                class_ids=self._class_ids,
            ),
            frame=frame,
        )

    def reset(self) -> None:
        self._tracking_command.reset()
