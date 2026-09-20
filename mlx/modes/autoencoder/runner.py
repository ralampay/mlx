from __future__ import annotations

from typing import Any

from mlx.core.commands import NullWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.commands import (
    EmbedAutoencoder,
    ListAutoencoderLosses,
    ListAutoencoderModels,
    TrainAutoencoder,
)
from mlx.modes.autoencoder.presentation import RichAutoencoderReporter, display_inventory
from mlx.modes.autoencoder.requests import AutoencoderEmbedRequest, AutoencoderTrainRequest


def _reporter(config):
    return (
        NullWorkflowReporter()
        if config.get("output_format") == "json"
        else RichAutoencoderReporter()
    )


def _train(config: dict[str, Any]):
    values = _mode_defaults(config, training=True)
    return TrainAutoencoder(
        AutoencoderTrainRequest.from_config(values), reporter=_reporter(config)
    ).execute()


def _embed(config: dict[str, Any]):
    values = _mode_defaults(config, training=False)
    if "model" not in set(config.get("_explicit_options") or ()):
        values["model"] = None
    return EmbedAutoencoder(
        AutoencoderEmbedRequest.from_config(values), reporter=_reporter(config)
    ).execute()


def _mode_defaults(config: dict[str, Any], *, training: bool) -> dict[str, Any]:
    values = dict(config)
    explicit = set(config.get("_explicit_options") or ())
    defaults = {
        "batch_size": 64,
        "workers": 0,
        "random_seed": 42,
        "hidden_dim": 256,
        "bottleneck_dim": 128,
    }
    if training:
        defaults.update({"epochs": 50, "lr": 0.001, "val_ratio": 0.2})
    for name, value in defaults.items():
        if name not in explicit:
            values[name] = value
    return values


def _list_models(config: dict[str, Any]):
    values = ListAutoencoderModels().execute()
    if config.get("output_format") != "json":
        display_inventory("Autoencoder Models", values)
    return values


def _list_losses(config: dict[str, Any]):
    values = ListAutoencoderLosses().execute()
    if config.get("output_format") != "json":
        display_inventory("Autoencoder Loss Functions", values)
    return values


ACTION_HANDLERS = {
    "embed": _embed,
    "ls-loss-functions": _list_losses,
    "ls-models": _list_models,
    "train": _train,
}


def run_autoencoder(config: dict[str, Any]):
    action = config.get("action") or "ls-models"
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        raise MLXUserError(
            f"Unsupported action '{action}' for autoencoder. Available actions: "
            + ", ".join(sorted(ACTION_HANDLERS))
            + "."
        )
    return handler(config)


__all__ = ["run_autoencoder"]
