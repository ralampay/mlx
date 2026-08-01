from typing import Any

from mlx.modes.object_detection.ultralytics.tracking_adapter import (
    RunObjectDetectionTrackingCommand,
    to_tracking_detections,
)


def run_object_detection(config: dict[str, Any]) -> Any:
    """Load the optional Ultralytics runtime only when its CLI mode is invoked."""

    from mlx.modes.object_detection.ultralytics.runner import (
        run_object_detection as _run_object_detection,
    )

    return _run_object_detection(config)


__all__ = [
    "RunObjectDetectionTrackingCommand",
    "run_object_detection",
    "to_tracking_detections",
]
