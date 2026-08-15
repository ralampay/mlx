from __future__ import annotations

from typing import Any

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
from mlx.modes.segmentation.presentation import print_segmentation_config_summary
from mlx.modes.segmentation.requests import (
    BuildSegmentationDatasetRequest,
    SegmentationRequest,
)
from mlx.modes.segmentation.train import (
    SmokeTestSegmentationModel,
    TrainSegmentationModel,
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


ACTION_HANDLERS = {
    "benchmark": lambda config: BenchmarkSegmentation(
        SegmentationRequest.from_config(config)
    ).execute(),
    "build-dataset": lambda config: BuildSegmentationDataset(
        BuildSegmentationDatasetRequest.from_config(config)
    ).execute(),
    "infer-camera": lambda config: RunSegmentationStreamInference(
        SegmentationRequest.from_config(config), source="camera"
    ).execute(),
    "infer-image": lambda config: InferSegmentationImage(
        SegmentationRequest.from_config(config)
    ).execute(),
    "infer-video": lambda config: RunSegmentationStreamInference(
        SegmentationRequest.from_config(config), source="video"
    ).execute(),
    "ls-models": _list_models,
    "test": lambda config: SmokeTestSegmentationModel(
        SegmentationRequest.from_config(config)
    ).execute(),
    "train": lambda config: TrainSegmentationModel(
        SegmentationRequest.from_config(config)
    ).execute(),
}


def run_segmentation(mode_config: dict[str, Any]) -> Any:
    config = {**DEFAULT_CONFIG, **mode_config}
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
