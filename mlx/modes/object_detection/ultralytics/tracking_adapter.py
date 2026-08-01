from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.tracking import BoundingBox, TrackingDetection
from mlx.modes.object_detection.tracking.command import (
    RunTrackByDetectionCommand,
    validate_tracking_frame,
)
from mlx.modes.object_detection.tracking.models import TrackingFrameResult
from mlx.modes.object_detection.tracking.protocols import TrackingAlgorithm
from mlx.modes.object_detection.ultralytics.results import Detection, DetectionAdapter


def to_tracking_detections(
    detections: Sequence[Detection],
) -> tuple[TrackingDetection, ...]:
    """Convert normalized detector output without retaining Ultralytics results."""

    return tuple(
        TrackingDetection(
            bounding_box=BoundingBox(*detection.xyxy),
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
        )
        for detection in detections
    )


class RunObjectDetectionTrackingCommand:
    """Run the project's normalized detection model and tracker for one frame.

    The injected detection model may be either current project adapter:
    ``UltralyticsDetectionAdapter`` for ``.pt`` models or
    ``OnnxRuntimeDetectionAdapter`` for ``.onnx`` models. Detector-specific output is
    converted here, while the injected tracking algorithm sees only generic tracking
    detections.
    """

    def __init__(
        self,
        *,
        detection_model: DetectionAdapter,
        algorithm: TrackingAlgorithm,
    ) -> None:
        self._detection_model = detection_model
        self._tracking_command = RunTrackByDetectionCommand(algorithm=algorithm)

    @property
    def frame_index(self) -> int:
        return self._tracking_command.frame_index

    def execute(self, *, frame: np.ndarray) -> TrackingFrameResult:
        validate_tracking_frame(frame)
        detection_result = self._detection_model.predict(frame)
        return self._tracking_command.execute(
            detections=to_tracking_detections(detection_result.detections),
            frame=frame,
        )

    def reset(self) -> None:
        self._tracking_command.reset()
