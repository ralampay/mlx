"""Lazy construction of the provider's supported scratch architectures."""

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.libreyolo.utils import (
    LibreYOLOModelSpec,
    build_drax_config,
    dependency_error,
)


def build_scratch_model(spec: LibreYOLOModelSpec, *, device: str) -> Any:
    try:
        import libreyolo
    except ImportError as exc:
        raise dependency_error("constructing an object-detection model") from exc

    try:
        constructor = getattr(libreyolo, spec.constructor_name)
    except (AttributeError, ImportError) as exc:
        raise MLXUserError(
            f"The installed LibreYOLO does not expose {spec.constructor_name}. "
            "Update the ralampay/libreyolo release dependency with './update.sh' "
            "from the MLX repository and try again."
        ) from exc

    kwargs = {"model_path": None, "size": spec.size, "device": device, "task": "detect"}
    if spec.uses_drax:
        kwargs["drax_config"] = build_drax_config(spec)
    return constructor(**kwargs)
