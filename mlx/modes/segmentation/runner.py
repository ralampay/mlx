from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.data import build_segmentation_dataset
from mlx.modes.segmentation.inference import (
    StreamSegmentationInferenceRunner,
    infer_segmentation_image,
)
from mlx.modes.segmentation.models import DEFAULT_MODEL
from mlx.modes.segmentation.presentation import print_segmentation_config_summary
from mlx.modes.segmentation.train import smoke_test_segmentation, train_segmentation

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
}

ACTION_HANDLERS = {
    "build-dataset": lambda config: build_segmentation_dataset(config["dataset_path"]),
    "infer-camera": lambda config: StreamSegmentationInferenceRunner(
        config, source="camera"
    ).execute(),
    "infer-image": infer_segmentation_image,
    "infer-video": lambda config: StreamSegmentationInferenceRunner(
        config, source="video"
    ).execute(),
    "test": smoke_test_segmentation,
    "train": train_segmentation,
}


def run_segmentation(mode_config: dict[str, Any]) -> Any:
    config = {**DEFAULT_CONFIG, **mode_config}
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
