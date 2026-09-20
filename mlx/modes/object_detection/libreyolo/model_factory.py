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


def validate_incremental_adapter_support(model) -> None:
    """Reject unsupported kwargs before a provider can silently ignore them."""
    from dataclasses import fields, is_dataclass
    from inspect import signature

    required = {"incremental_adapter", "incremental_adapter_train_only", "incremental_adapter_type"}
    parameters = signature(model.train).parameters
    if required <= parameters.keys():
        return
    # LibreYOLO's YOLOX train(**kwargs) forwards options to the trainer's config.
    trainer_factory = getattr(model, "_trainer_class", None)
    if callable(trainer_factory):
        trainer = trainer_factory()
        config_factory = getattr(trainer, "_config_class", None)
        config = config_factory() if callable(config_factory) else None
        if config is not None and is_dataclass(config):
            if required <= {item.name for item in fields(config)}:
                return
    raise MLXUserError(
        "The installed LibreYOLO model does not expose incremental-adapter training options. "
        "Use a provider release with explicit adapter support, or omit --incremental-adapter. "
        "MLX will not silently run full-model training instead."
    )
