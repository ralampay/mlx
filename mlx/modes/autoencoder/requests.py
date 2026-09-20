from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class AutoencoderRequest(ConfigRequest):
    model: Optional[str] = "simple"
    model_path: Optional[str] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    input_dim: Optional[int] = None
    hidden_dim: int = 256
    bottleneck_dim: int = 128
    batch_size: int = 64
    workers: int = 0
    device: str = "cpu"
    random_seed: Optional[int] = 42
    normalize_inputs: Optional[bool] = None
    autoencoder_config: Optional[str] = None


@dataclass(frozen=True)
class AutoencoderTrainRequest(AutoencoderRequest):
    epochs: int = 50
    lr: Optional[float] = 0.001
    val_ratio: Optional[float] = 0.2
    loss: str = "mse"
    loss_config: Optional[str] = None
    plots: bool = True
    use_best: bool = True


@dataclass(frozen=True)
class AutoencoderEmbedRequest(AutoencoderRequest):
    normalize_embeddings: bool = False


__all__ = [
    "AutoencoderEmbedRequest",
    "AutoencoderRequest",
    "AutoencoderTrainRequest",
]
