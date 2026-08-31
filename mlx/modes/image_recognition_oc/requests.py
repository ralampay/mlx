from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class ImageOneClassRequest(ConfigRequest):
    model: Optional[str] = "deep-svdd"
    backbone: Optional[str] = "resnet18"
    model_path: Optional[str] = None
    dataset_path: str = ""
    dataset_s3_uri: Optional[str] = None
    dataset_cache_dir: str = "~/.cache/mlx/datasets"
    output_path: Optional[str] = None
    input_img: Optional[str] = None
    device: str = "cpu"
    width: int = 224
    height: int = 224
    batch_size: int = 16
    workers: int = 0
    colored: bool = True
    pretrained: bool = False
    random_seed: Optional[int] = None
    drax_fusion_mode: str = "average"
    svdd_dim: int = 128
    svdd_hidden_dim: int = 256
    svdd_quantile: float = 0.95


@dataclass(frozen=True)
class TrainImageOneClassRequest(ImageOneClassRequest):
    epochs: int = 50
    lr: Optional[float] = 0.001
    use_best: bool = True
    apply_transformations: bool = False


@dataclass(frozen=True)
class InferImageOneClassRequest(ImageOneClassRequest):
    """Inputs for calibrated single-image one-class inference."""


@dataclass(frozen=True)
class BenchmarkImageOneClassRequest(ImageOneClassRequest):
    plots: bool = True


@dataclass(frozen=True)
class ListImageOneClassModelsRequest(ConfigRequest):
    model: Optional[str] = None
    backbone: Optional[str] = None
    colored: bool = True
    drax_fusion_mode: str = "average"
    svdd_dim: int = 128
    svdd_hidden_dim: int = 256


__all__ = [
    "BenchmarkImageOneClassRequest",
    "ImageOneClassRequest",
    "InferImageOneClassRequest",
    "ListImageOneClassModelsRequest",
    "TrainImageOneClassRequest",
]
