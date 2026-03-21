from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.ultralytics.inference import StreamInferenceRunner
from mlx.modes.object_detection.ultralytics.training import train_object_detection


def run_object_detection(config: dict[str, Any]) -> Any:
    action = config.get("action", "train")
    if action == "train":
        return train_object_detection(config)
    if action == "infer-camera":
        return StreamInferenceRunner(config, source="camera").execute()
    if action == "infer-video":
        return StreamInferenceRunner(config, source="video").execute()

    raise MLXUserError(
        "Unsupported action "
        f"'{action}' for object-detection. Supported actions: train, infer-camera, infer-video."
    )
