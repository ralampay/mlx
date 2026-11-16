"""CLI-independent requests and validation for temporal generation."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.requests import ConfigRequest


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MLXUserError(f"{name} must be an integer >= {minimum}.")
    return value


def finite_number(value, name, minimum=0, strict=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (strict and value == minimum)
    ):
        raise MLXUserError(
            f"{name} must be finite and {'>' if strict else '>='} {minimum}."
        )
    return value


@dataclass(frozen=True)
class TrainRequest(ConfigRequest):
    input_path: str | None = None
    output_path: str | None = None
    time_column: str | None = None
    feature_columns: tuple[str, ...] = ()
    series_id_column: str | None = None
    window_size: int = 128
    stride: int = 32
    model: str = "timevae"
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (64, 128)
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.001
    kl_weight: float = 1.0
    random_seed: int = 42
    device: str = "cpu"
    model_config: Mapping[str, Any] | None = None

    def validate(self):
        if self.model_config is not None and not isinstance(self.model_config, Mapping):
            raise MLXUserError(
                "model_config must be a mapping of model-specific options."
            )
        positive_int(self.window_size, "window-size", 4)
        for name in ("stride", "latent_dim", "epochs", "batch_size"):
            positive_int(getattr(self, name), name)
        positive_int(self.random_seed, "seed", 0)
        if not isinstance(self.hidden_dims, (tuple, list)) or not self.hidden_dims:
            raise MLXUserError("hidden-dims must contain at least one positive width.")
        for width in self.hidden_dims:
            positive_int(width, "hidden-dims")
        finite_number(self.lr, "lr", strict=True)
        finite_number(self.kl_weight, "kl-weight")


@dataclass(frozen=True)
class SynthesizeRequest(ConfigRequest):
    model_path: str | None = None
    output_path: str | None = None
    num_sequences: int = 1000
    batch_size: int = 64
    random_seed: int = 123
    device: str = "cpu"


@dataclass(frozen=True)
class BenchmarkRequest(ConfigRequest):
    model_path: str | None = None
    input_path: str | None = None
    synthetic_data: str | None = None
    output_path: str | None = None
    random_seed: int = 42
    bootstrap_samples: int = 100
    max_metric_windows: int = 512
