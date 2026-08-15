from __future__ import annotations

import hashlib

import cv2
import numpy as np

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import print_error, print_info, print_success, print_warning
from mlx.modes.object_detection.models import DetectionResult


class RichWorkflowReporter:
    def emit(self, event: WorkflowEvent) -> None:
        if event.level == "error":
            print_error(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        elif event.level == "success":
            print_success(event.message)
        else:
            print_info(event.message)


def annotate_detections(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    annotated = frame.copy()
    for detection in result.detections:
        x1, y1, x2, y2 = detection.xyxy
        color = _color_for_label(detection.label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{detection.label}: {detection.confidence:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return tuple(int(min(max(int(digest[index : index + 2], 16), 64), 255)) for index in (0, 2, 4))

