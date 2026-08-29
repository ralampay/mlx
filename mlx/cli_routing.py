from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


class UnknownModeError(ValueError):
    """Raised when the selected mode is not registered."""


ModeRunner = Callable[[dict[str, Any]], Any]

MODE_REGISTRY: dict[str, str] = {
    "image-classification": "mlx.modes.image_classification.runner:run_image_classification",
    "image_classification": "mlx.modes.image_classification.runner:run_image_classification",
    "object-detection": "mlx.modes.object_detection.runner:run_object_detection",
    "object_detection": "mlx.modes.object_detection.runner:run_object_detection",
    "track": "mlx.modes.object_detection.tracking.runner:run_tracking",
    "tracking": "mlx.modes.object_detection.tracking.runner:run_tracking",
    "segmentation": "mlx.modes.segmentation.runner:run_segmentation",
    "video-anomaly-detection": "mlx.modes.video_anomaly_detection.runner:run_video_anomaly_detection",
    "video_anomaly_detection": "mlx.modes.video_anomaly_detection.runner:run_video_anomaly_detection",
    "nlp": "mlx.modes.nlp.runner:run_nlp",
}


def resolve_mode_runner(mode: str) -> ModeRunner:
    dotted_path = MODE_REGISTRY.get(mode)
    if dotted_path is None:
        raise UnknownModeError(f"Unknown mode '{mode}'.")
    module_path, function_name = dotted_path.split(":")
    return getattr(import_module(module_path), function_name)


__all__ = ["MODE_REGISTRY", "ModeRunner", "UnknownModeError", "resolve_mode_runner"]
