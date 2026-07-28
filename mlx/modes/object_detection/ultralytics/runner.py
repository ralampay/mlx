from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.object_detection.ultralytics.conversion import convert_object_detection_model
from mlx.modes.object_detection.ultralytics.inference import StreamInferenceRunner
from mlx.modes.object_detection.ultralytics.list_models import ListObjectDetectionModels
from mlx.modes.object_detection.ultralytics.training import train_object_detection


def _list_models():
    summaries = ListObjectDetectionModels().execute()
    print_model_parameter_table(summaries, title="Object Detection Models")
    return summaries


def run_object_detection(config: dict[str, Any]) -> Any:
    action = config.get("action", "train")
    if action == "train":
        return train_object_detection(config)
    if action == "infer-camera":
        return StreamInferenceRunner(config, source="camera").execute()
    if action == "infer-video":
        return StreamInferenceRunner(config, source="video").execute()
    if action == "convert":
        return convert_object_detection_model(config)
    if action == "ls-models":
        return _list_models()

    raise MLXUserError(
        "Unsupported action "
        f"'{action}' for object-detection. Supported actions: train, infer-camera, "
        "infer-video, convert, ls-models."
    )
