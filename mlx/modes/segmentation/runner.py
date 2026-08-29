from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.datasets import (
    TrainWithDatasetSource,
    segmentation_dataset_root,
    validate_dataset_source_options,
)
from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.segmentation.data import BuildSegmentationDataset
from mlx.modes.segmentation.evaluation import BenchmarkSegmentation
from mlx.modes.segmentation.inference import (
    InferSegmentationImage,
    RunSegmentationStreamInference,
)
from mlx.modes.segmentation.list_models import ListSegmentationModels
from mlx.modes.segmentation.models import DEFAULT_MODEL
from mlx.modes.segmentation.presentation import (
    display_segmentation_result,
    print_segmentation_config_summary,
    RichSegmentationReporter,
    resolve_segmentation_dataset_build_request,
)
from mlx.modes.segmentation.requests import (
    BuildSegmentationDatasetRequest,
    SegmentationRequest,
)
from mlx.modes.segmentation.train import (
    SmokeTestSegmentationModel,
    TrainSegmentationModel,
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
    "lr": None,
    "mask_threshold": 0.5,
    "num_classes": 2,
    "overlay_alpha": 0.45,
    "split": "test",
}


def _list_models(config: dict[str, Any]):
    summaries = ListSegmentationModels(config).execute()
    print_model_parameter_table(summaries, title="Segmentation Models")
    return summaries


def _infer_image(config: dict[str, Any]):
    result = InferSegmentationImage(
        SegmentationRequest.from_config(config)
    ).execute()
    if config.get("display", True):
        display_segmentation_result(result)
    return result


def _run_stream(config: dict[str, Any], *, source: str):
    request = SegmentationRequest.from_config(config)
    frame_source = OpenCVSegmentationFrameSource(
        source=source,
        camera_index=request.camera_index,
        file_path=request.file_path,
    )
    frame_sink = OpenCVSegmentationFrameSink(
        title=(
            "MLX Segmentation (Camera)"
            if source == "camera"
            else "MLX Segmentation (Video)"
        ),
        delay_ms=1 if source == "camera" else 10,
    )
    return RunSegmentationStreamInference(
        request,
        source=source,
        frame_source=frame_source,
        frame_sink=frame_sink,
        reporter=RichSegmentationReporter(),
    ).execute()


def _train(config: dict[str, Any]):
    request = SegmentationRequest.from_config(config)
    reporter = RichSegmentationReporter()
    return TrainWithDatasetSource(
        request,
        trainer_factory=lambda resolved: TrainSegmentationModel(
            resolved, reporter=reporter
        ),
        root_resolver=segmentation_dataset_root,
        artifact_dir_resolver=lambda resolved: resolve_train_output_paths(
            resolved.to_config(), model_name=resolve_model_name(resolved.to_config())
        )["output_dir"],
        profile=config.get("profile"),
        reporter=reporter,
    ).execute()


ACTION_HANDLERS = {
    "benchmark": lambda config: BenchmarkSegmentation(
        SegmentationRequest.from_config(config),
        reporter=RichSegmentationReporter(),
    ).execute(),
    "build-dataset": lambda config: BuildSegmentationDataset(
        BuildSegmentationDatasetRequest.from_config(config),
        reporter=RichSegmentationReporter(),
        input_resolver=resolve_segmentation_dataset_build_request,
    ).execute(),
    "infer-camera": lambda config: _run_stream(config, source="camera"),
    "infer-image": _infer_image,
    "infer-video": lambda config: _run_stream(config, source="video"),
    "ls-models": _list_models,
    "test": lambda config: SmokeTestSegmentationModel(
        SegmentationRequest.from_config(config),
        reporter=RichSegmentationReporter(),
    ).execute(),
    "train": _train,
}


def run_segmentation(mode_config: dict[str, Any]) -> Any:
    config = {**DEFAULT_CONFIG, **mode_config}
    validate_dataset_source_options(config, action=config["action"])
    if config["action"] == "ls-models":
        return ACTION_HANDLERS["ls-models"](config)

    config["model"] = mode_config.get("model") or DEFAULT_MODEL
    config["input_size"] = tuple(config.get("input_size", (config["width"], config["height"])))

    print_segmentation_config_summary(config["model"], config)

    handler = ACTION_HANDLERS.get(config["action"])
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{config['action']}' for segmentation. Available actions: {available}."
        )
    return handler(config)
