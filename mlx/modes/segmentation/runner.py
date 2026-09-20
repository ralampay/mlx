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
from mlx.modes.segmentation.data import (
    BuildSegmentationDataset,
    normalize_segmentation_transform,
    segmentation_dataset_root,
)
from mlx.modes.segmentation.evaluation import BenchmarkSegmentation
from mlx.modes.segmentation.inference import (
    InferSegmentationImage,
    RunSegmentationStreamInference,
)
from mlx.modes.segmentation.list_models import ListSegmentationModels
from mlx.modes.segmentation.models import DEFAULT_MODEL, MODEL_GROUP_NAMES
from mlx.modes.segmentation.presentation import (
    display_segmentation_result,
    print_segmentation_config_summary,
    RichSegmentationReporter,
    resolve_segmentation_dataset_build_request,
)
from mlx.modes.segmentation.requests import (
    BuildSegmentationDatasetRequest,
    BenchmarkSegmentationRequest,
    InferSegmentationRequest,
    SegmentationRequest,
    SmokeTestSegmentationRequest,
    TrainSegmentationRequest,
)
from mlx.modes.segmentation.train import (
    SmokeTestSegmentationModel,
    TrainSegmentationModel,
)
from mlx.modes.segmentation.train_all import (
    TrainAllSegmentationModels,
    validate_all_models_request,
)
from mlx.modes.segmentation.utils import resolve_model_name, resolve_train_output_paths
from mlx.modes.segmentation.streaming import (
    OpenCVSegmentationFrameSink,
    OpenCVSegmentationFrameSource,
)

DEFAULT_CONFIG = {
    "action": "test",
    "batch_size": 4,
    "colored": True,
    "dataset_path": "",
    "device": "cpu",
    "epochs": 50,
    "input_size": (256, 256),
    "transform": "resize",
    "lr": None,
    "mask_threshold": 0.5,
    "num_classes": 2,
    "overlay_alpha": 0.45,
    "split": "test",
}


def _reporter(config: dict[str, Any]):
    return NullWorkflowReporter() if config.get("output_format") == "json" else RichSegmentationReporter()


def _list_models(config: dict[str, Any]):
    if config.get("names_only"):
        from mlx.modes.segmentation.models import supported_model_names
        from mlx.core.model_listing import ListComponentNames
        from mlx.core.presentation import display_component_inventory
        result = ListComponentNames(supported_model_names()).execute()
        if config.get("output_format") != "json":
            display_component_inventory(result, title="Available Models")
        return result

    summaries = ListSegmentationModels(config).execute()
    if config.get("output_format") != "json":
        print_model_parameter_table(summaries, title="Segmentation Models")
    return summaries


def _infer_image(config: dict[str, Any]):
    result = InferSegmentationImage(
        InferSegmentationRequest.from_config(config)
    ).execute()
    if config.get("output_format") != "json" and config.get("display", True):
        display_segmentation_result(result)
    return result


def _run_stream(config: dict[str, Any], *, source: str):
    request = InferSegmentationRequest.from_config(config)
    frame_source = OpenCVSegmentationFrameSource(
        source=source,
        camera_index=request.camera_index,
        file_path=request.file_path,
    )
    frame_sink = (
        NullFrameSink()
        if config.get("output_format") == "json"
        else OpenCVSegmentationFrameSink(
            title=(
                "MLX Segmentation (Camera)"
                if source == "camera"
                else "MLX Segmentation (Video)"
            ),
            delay_ms=1 if source == "camera" else 10,
        )
    )
    return RunSegmentationStreamInference(
        request,
        source=source,
        frame_source=frame_source,
        frame_sink=frame_sink,
        reporter=_reporter(config),
    ).execute()


def _train(config: dict[str, Any]):
    request_config = dict(config)
    explicit = set(config.get("_explicit_options") or ())
    if request_config.get("dataset_s3_uri") and "dataset_path" not in explicit:
        request_config["dataset_path"] = ""
    request = TrainSegmentationRequest.from_config(request_config)
    reporter = _reporter(config)
    train_all = request.model in MODEL_GROUP_NAMES
    if train_all:
        validate_all_models_request(request)

    def trainer_factory(resolved: TrainSegmentationRequest):
        if train_all:
            return TrainAllSegmentationModels(
                resolved,
                reporter=reporter,
                allow_dataset_source_manifest=bool(resolved.dataset_s3_uri),
            )
        return TrainSegmentationModel(resolved, reporter=reporter)

    def artifact_dir_resolver(resolved: TrainSegmentationRequest) -> Path:
        if train_all:
            return Path(str(resolved.output_path)).expanduser()
        return resolve_train_output_paths(
            resolved.to_config(), model_name=resolve_model_name(resolved.to_config())
        )["output_dir"]

    return TrainWithDatasetSource(
        request,
        trainer_factory=trainer_factory,
        root_resolver=segmentation_dataset_root,
        artifact_dir_resolver=artifact_dir_resolver,
        profile=config.get("profile"),
        reporter=reporter,
    ).execute()


ACTION_HANDLERS = {
    "benchmark": lambda config: BenchmarkSegmentation(
        BenchmarkSegmentationRequest.from_config(config),
        reporter=_reporter(config),
    ).execute(),
    "build-dataset": lambda config: BuildSegmentationDataset(
        BuildSegmentationDatasetRequest.from_config(config),
        reporter=_reporter(config),
        input_resolver=resolve_segmentation_dataset_build_request,
    ).execute(),
    "infer-camera": lambda config: _run_stream(config, source="camera"),
    "infer-image": _infer_image,
    "infer-video": lambda config: _run_stream(config, source="video"),
    "ls-models": _list_models,
    "test": lambda config: SmokeTestSegmentationModel(
        SmokeTestSegmentationRequest.from_config(config),
        reporter=_reporter(config),
    ).execute(),
    "train": _train,
}


def run_segmentation(mode_config: dict[str, Any]) -> Any:
    config = {**DEFAULT_CONFIG, **mode_config}
    config["action"] = config.get("action") or DEFAULT_CONFIG["action"]
    validate_dataset_source_options(config, action=config["action"])
    if config["action"] == "ls-models":
        return ACTION_HANDLERS["ls-models"](config)

    config["model"] = mode_config.get("model") or DEFAULT_MODEL
    if config["model"] in MODEL_GROUP_NAMES and config["action"] != "train":
        raise MLXUserError(
            f"Segmentation --model {config['model']} is supported only with --action train."
        )
    config["input_size"] = tuple(config.get("input_size", (config["width"], config["height"])))
    config["transform"] = normalize_segmentation_transform(config.get("transform"))

    if config.get("output_format") != "json":
        print_segmentation_config_summary(config["model"], config)

    handler = ACTION_HANDLERS.get(config["action"])
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{config['action']}' for segmentation. Available actions: {available}."
        )
    return handler(config)
