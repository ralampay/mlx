from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.object_detection.commands import (
    BenchmarkObjectDetectionModel,
    ConvertObjectDetectionModel,
    CreateObjectDetector,
    ListObjectDetectionModels,
    RunObjectDetectionStream,
    TrainObjectDetectionModel,
)
from mlx.modes.object_detection.presentation import (
    RichWorkflowReporter,
    annotate_detections,
    print_benchmark_result,
)
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    StreamObjectDetectionRequest,
    TrainObjectDetectionRequest,
)
from mlx.modes.object_detection.streaming import OpenCVFrameSink, OpenCVFrameSource


def run_object_detection(config: dict[str, Any]) -> Any:
    if config.get("platform", "local") == "aws":
        from mlx.modes.object_detection.aws.runner import run_aws_object_detection

        return run_aws_object_detection(config)

    action = config.get("action") or "train"
    reporter = RichWorkflowReporter()

    if action == "train":
        result = TrainObjectDetectionModel(
            TrainObjectDetectionRequest.from_config(config),
            reporter=reporter,
        ).execute()
        benchmark_result = getattr(result, "benchmark_result", None)
        if benchmark_result is not None:
            print_benchmark_result(benchmark_result)
        return result
    if action == "benchmark":
        benchmark_config = dict(config)
        explicit = set(config.get("_explicit_options") or ())
        if "confidence" not in explicit:
            benchmark_config["confidence"] = 0.001
        if "batch_size" not in explicit:
            benchmark_config["batch_size"] = 16
        if "height" not in explicit:
            benchmark_config["height"] = 640
        if "width" not in explicit:
            benchmark_config["width"] = 640
        result = BenchmarkObjectDetectionModel(
            BenchmarkObjectDetectionRequest.from_config(benchmark_config),
            reporter=reporter,
        ).execute()
        print_benchmark_result(result)
        return result
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

    available = "benchmark, convert, infer-camera, infer-video, ls-models, train"
    raise MLXUserError(
        f"Unsupported action '{action}' for object-detection. Available actions: {available}."
    )
