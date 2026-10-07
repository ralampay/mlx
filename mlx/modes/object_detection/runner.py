from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from mlx.core.datasets import (
    TrainWithDatasetSource,
    validate_dataset_source_options,
)
from mlx.core.commands import NullWorkflowReporter
from mlx.core.streaming import NullFrameSink
from mlx.core.exceptions import MLXUserError
from mlx.core.configuration import with_explicit_options
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
    config = with_explicit_options(config)
    from mlx.modes.object_detection.distillation import validate_distillation_options
    validate_distillation_options(config)
    if config.get("action") == "adapter-prepare-neu-det":
        if config.get("platform", "local") != "local":
            raise MLXUserError("NEU-DET preparation is local only")
        from mlx.modes.object_detection.neu_det import PrepareNeuDetDataset
        return PrepareNeuDetDataset(Path(config.get("dataset_path") or "~/Desktop/datasets/neu-det")).execute()
    if config.get("action") in {"adapter-zero-shot", "adapter-zero-shot-report"}:
        if config.get("platform", "local") != "local":
            raise MLXUserError("Adapter zero-shot studies run locally only.")
        from mlx.modes.object_detection.zero_shot.composition import run_transfer_action
        return run_transfer_action(config)
    if config.get("platform", "local") == "aws":
        from mlx.modes.object_detection.aws.runner import run_aws_object_detection

        return run_aws_object_detection(config)

    if config.get("model_s3_uri"):
        raise MLXUserError(
            "--model-s3-uri is supported only for AWS object-detection fine-tuning. "
            "Use --model-path for local workflows."
        )

    action = config.get("action") or "train"
    if action in {"adapter-verify", "adapter-prepare", "adapter-calibrate", "adapter-experiment", "adapter-report", "adapter-slice-predict", "adapter-slice-report"}:
        from mlx.modes.object_detection.adapter_experiment import (
            AdapterExperimentRequest, RunAdapterExperiment, DEFAULT_DATASET, DEFAULT_OUTPUT,
        )
        from mlx.modes.object_detection.libreyolo.adapter_backend import (
            VerifyFoundationCheckpoint,
        )
        from mlx.modes.object_detection.adapter_data import PrepareDawnAdapterDataset
        from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
        if action in {"adapter-slice-predict", "adapter-slice-report"}:
            from mlx.modes.object_detection.adapter_slices import (
                AdapterSliceRequest,
                CacheAdapterSlicePredictions,
                GenerateAdapterSliceReport,
            )
            slice_request = AdapterSliceRequest.from_config(config)
            command = (
                CacheAdapterSlicePredictions(slice_request)
                if action == "adapter-slice-predict"
                else GenerateAdapterSliceReport(slice_request)
            )
            return command.execute()
        if action == "adapter-report":
            return GenerateAdapterReport(
                Path(config.get("output_path") or DEFAULT_OUTPUT),
                baseline_study=config.get("baseline_study"),
                comparison_method=config.get("comparison_method") or "drax",
            ).execute()
        if action == "adapter-prepare":
            source = Path(config.get("dataset_path") or "~/Desktop/datasets/object-detection/dawn/original")
            destination = Path(config.get("output_path") or DEFAULT_DATASET)
            return PrepareDawnAdapterDataset(source, destination, seed=42).execute()
        request = AdapterExperimentRequest.from_config(config)
        if action == "adapter-verify":
            _, info = VerifyFoundationCheckpoint(request.model, request.checkpoint).execute()
            return info
        if action == "adapter-calibrate":
            from .libreyolo.adapter_execution import CalibrateAdapterExperiment
            return CalibrateAdapterExperiment(request).execute()
        from .adapter_composition import create_experiment_backend
        return RunAdapterExperiment(request, backend=create_experiment_backend()).execute()
    if action == "ls-models" and config.get("names_only"):
        from mlx.modes.object_detection.providers import get_provider
        from mlx.core.model_listing import ListComponentNames
        from mlx.core.presentation import display_component_inventory
        provider = get_provider(config.get("provider", "ultralytics"))
        names = getattr(provider, "model_names", None)
        if names is None:
            raise MLXUserError(f"Provider {provider.name} does not supply model-name metadata.")
        result = ListComponentNames(names()).execute()
        if config.get("output_format") != "json":
            display_component_inventory(result, title="Detection Models")
        return result
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
        if "verbose" not in explicit:
            benchmark_config["verbose"] = True
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
        with ExitStack() as setup:
            setup.callback(source.release)
            sink = (
                NullFrameSink()
                if is_json
                else OpenCVFrameSink(
                    title=f"MLX Object Detection ({stream_request.source.title()})",
                    delay_ms=1 if stream_request.source == "camera" else 10,
                )
            )
            setup.callback(sink.close)
            command = RunObjectDetectionStream(
                detector=detector,
                frame_source=source,
                frame_sink=sink,
                renderer=annotate_detections,
                reporter=reporter,
            )
            # The constructed command owns both ports during execution.
            setup.pop_all()
        return command.execute()

    available = "adapter-calibrate, adapter-experiment, adapter-prepare, adapter-report, adapter-slice-predict, adapter-slice-report, adapter-verify, benchmark, convert, fine-tune, infer-camera, infer-video, ls-models, train"
    raise MLXUserError(
        f"Unsupported action '{action}' for object-detection. Available actions: {available}."
    )
