from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.datasets import (
    TrainWithDatasetSource,
    validate_dataset_source_options,
)
from mlx.core.commands import NullWorkflowReporter
from mlx.core.streaming import NullFrameSink
from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.object_detection.commands import (
    BenchmarkObjectDetectionModel,
    ConvertObjectDetectionModel,
    CreateObjectDetector,
    FineTuneObjectDetectionModel,
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
    FineTuneObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    StreamObjectDetectionRequest,
    TrainObjectDetectionRequest,
)
from mlx.modes.object_detection.streaming import OpenCVFrameSink, OpenCVFrameSource
from mlx.modes.object_detection.data import object_detection_dataset_root


def run_object_detection(config: dict[str, Any]) -> Any:
    if config.get("platform", "local") == "aws":
        from mlx.modes.object_detection.aws.runner import run_aws_object_detection

        return run_aws_object_detection(config)

    if config.get("model_s3_uri"):
        raise MLXUserError(
            "--model-s3-uri is supported only for AWS object-detection fine-tuning. "
            "Use --model-path for local workflows."
        )

    action = config.get("action") or "train"
    is_json = config.get("output_format") == "json"
    reporter = NullWorkflowReporter() if is_json else RichWorkflowReporter()

    if action in {"train", "fine-tune"}:
        validate_dataset_source_options(config, action=action)
        request_type = (
            FineTuneObjectDetectionRequest
            if action == "fine-tune"
            else TrainObjectDetectionRequest
        )
        command_type = (
            FineTuneObjectDetectionModel
            if action == "fine-tune"
            else TrainObjectDetectionModel
        )
        request = request_type.from_config(config)
        result = TrainWithDatasetSource(
            request,
            trainer_factory=lambda resolved: command_type(
                resolved, reporter=reporter
            ),
            root_resolver=object_detection_dataset_root,
            artifact_dir_resolver=lambda resolved: Path(str(resolved.output_path)),
            profile=config.get("profile"),
            reporter=reporter,
        ).execute()
        benchmark_result = getattr(result, "benchmark_result", None)
        if benchmark_result is not None and not is_json:
            print_benchmark_result(benchmark_result)
        return result
    validate_dataset_source_options(config, action=action)
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
        if not is_json:
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
        if not is_json:
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
        sink = (
            NullFrameSink()
            if is_json
            else OpenCVFrameSink(
                title=f"MLX Object Detection ({stream_request.source.title()})",
                delay_ms=1 if stream_request.source == "camera" else 10,
            )
        )
        return RunObjectDetectionStream(
            detector=detector,
            frame_source=source,
            frame_sink=sink,
            renderer=annotate_detections,
            reporter=reporter,
        ).execute()

    available = "benchmark, convert, fine-tune, infer-camera, infer-video, ls-models, train"
    raise MLXUserError(
        f"Unsupported action '{action}' for object-detection. Available actions: {available}."
    )
