"""Compatibility re-exports for provider-neutral tracking composition."""

from mlx.modes.object_detection.tracking_adapter import (
    RunObjectDetectionTrackingCommand,
    to_tracking_detections,
)

__all__ = ["RunObjectDetectionTrackingCommand", "to_tracking_detections"]
