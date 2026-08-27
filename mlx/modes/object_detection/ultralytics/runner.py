from __future__ import annotations

from typing import Any


def _list_models():
    """Keep the legacy extension seam while using the neutral command."""

    from mlx.core.ui import print_model_parameter_table
    from mlx.modes.object_detection.commands import ListObjectDetectionModels
    from mlx.modes.object_detection.requests import ListObjectDetectionModelsRequest

    summaries = ListObjectDetectionModels(
        ListObjectDetectionModelsRequest(provider="ultralytics")
    ).execute()
    print_model_parameter_table(summaries, title="Object Detection Models")
    return summaries


def run_object_detection(config: dict[str, Any]) -> Any:
    """Compatibility composition root for the former Ultralytics-only runner."""

    if config.get("action", "train") == "ls-models":
        return _list_models()

    from mlx.modes.object_detection.runner import run_object_detection as run_provider_neutral

    normalized = dict(config)
    normalized["provider"] = "ultralytics"
    return run_provider_neutral(normalized)


__all__ = ["run_object_detection"]
