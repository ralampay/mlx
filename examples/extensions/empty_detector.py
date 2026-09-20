from mlx.modes.object_detection.models import DetectionResult


class EmptyDetector:
    """An inference adapter with no detections, useful for testing plumbing."""

    def predict(self, frame):
        return DetectionResult(detections=(), names={})
