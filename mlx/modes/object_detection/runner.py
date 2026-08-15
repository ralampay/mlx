from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.object_detection.commands import (
    ConvertObjectDetectionModel,
    CreateObjectDetector,
    ListObjectDetectionModels,
    RunObjectDetectionStream,
    TrainObjectDetectionModel,
)
from mlx.modes.object_detection.presentation import RichWorkflowReporter, annotate_detections
from mlx.modes.object_detection.requests import (
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    StreamObjectDetectionRequest,
    TrainObjectDetectionRequest,
)
from mlx.modes.object_detection.streaming import OpenCVFrameSink, OpenCVFrameSource


def run_object_detection(config: dict[str, Any]) -> Any:
    action = config.get("action") or "train"
    reporter = RichWorkflowReporter()

    if action == "train":
        return TrainObjectDetectionModel(
            TrainObjectDetectionRequest.from_config(config),
            reporter=reporter,
        ).execute()
    if action == "convert":
        return ConvertObjectDetectionModel(
            ConvertObjectDetectionRequest.from_config(config),
            reporter=reporter,
        ).execute()
    if action == "ls-models":
        summaries = ListObjectDetectionModels(
            ListObjectDetectionModelsRequest.from_config(config),
            reporter=reporter,
        ).execute()
        print_model_parameter_table(summaries, title="Object Detection Models")
        return summaries
    if action in {"infer-camera", "infer-video"}:
        stream_request = StreamObjectDetectionRequest.from_config(
            {**config, "source": "camera" if action == "infer-camera" else "video"}
        )
        detector = CreateObjectDetector(
            ObjectDetectionRequest.from_config(config)
        ).execute()
        source = OpenCVFrameSource(
            source=stream_request.source,
            camera_index=stream_request.camera_index,
            file_path=stream_request.file_path,
        )
        sink = OpenCVFrameSink(
            title=f"MLX Object Detection ({stream_request.source.title()})",
            delay_ms=1 if stream_request.source == "camera" else 10,
        )
        return RunObjectDetectionStream(
            detector=detector,
            frame_source=source,
            frame_sink=sink,
            renderer=annotate_detections,
            reporter=reporter,
        ).execute()

    available = "convert, infer-camera, infer-video, ls-models, train"
    raise MLXUserError(
        f"Unsupported action '{action}' for object-detection. Available actions: {available}."
    )

