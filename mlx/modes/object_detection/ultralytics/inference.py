from __future__ import annotations

from typing import Any

from mlx.modes.object_detection.commands import (
    CreateObjectDetector,
    RunObjectDetectionStream,
)
from mlx.modes.object_detection.presentation import (
    RichWorkflowReporter,
    annotate_detections,
)
from mlx.modes.object_detection.requests import (
    ObjectDetectionRequest,
    StreamObjectDetectionRequest,
)
from mlx.modes.object_detection.streaming import OpenCVFrameSink, OpenCVFrameSource


class StreamInferenceRunner:
    """Backward-compatible adapter onto the provider-neutral stream command."""

    def __init__(self, config: dict[str, Any], source: str) -> None:
        self.config = {**config, "provider": "ultralytics", "source": source}
        self.source = source

    def execute(self):
        request = StreamObjectDetectionRequest.from_config(self.config)
        detector = CreateObjectDetector(
            ObjectDetectionRequest.from_config(self.config)
        ).execute()
        return RunObjectDetectionStream(
            detector=detector,
            frame_source=OpenCVFrameSource(
                source=request.source,
                camera_index=request.camera_index,
                file_path=request.file_path,
            ),
            frame_sink=OpenCVFrameSink(
                title=f"MLX Object Detection ({request.source.title()})",
                delay_ms=1 if request.source == "camera" else 10,
            ),
            renderer=annotate_detections,
            reporter=RichWorkflowReporter(),
        ).execute()


__all__ = ["StreamInferenceRunner"]
