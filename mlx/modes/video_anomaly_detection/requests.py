from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class VideoAnomalyRequest(ConfigRequest):
    model: Optional[str] = None
    model_path: Optional[str] = None
    dataset_path: str = ""
    output_path: Optional[str] = None
    device: str = "cpu"
    width: int = 224
    height: int = 224
    batch_size: int = 8
    workers: int = 0
    clip_length: int = 16
    frame_stride: int = 1
    backbone_mode: str = "3d"
    backbone_temporal_kernel_size: int = 3
    temporal_model: str = "tcn"
    temporal_hidden_dim: int = 256
    temporal_embedding_dim: int = 128
    temporal_kernel_size: int = 3
    temporal_dropout: float = 0.0
    svdd_dim: int = 128
    svdd_hidden_dim: int = 256
    svdd_quantile: float = 0.95
    pretrained: bool = False
    random_seed: Optional[int] = None
    drax_fusion_mode: str = "average"


@dataclass(frozen=True)
class TrainVideoAnomalyRequest(VideoAnomalyRequest):
    epochs: int = 50
    lr: Optional[float] = 0.001
    use_best: bool = True


@dataclass(frozen=True)
class BenchmarkVideoAnomalyRequest(VideoAnomalyRequest):
    frame_aggregation: str = "mean"
    plots: bool = True


@dataclass(frozen=True)
class InferVideoAnomalyRequest(VideoAnomalyRequest):
    file_path: Optional[str] = None


@dataclass(frozen=True)
class ListVideoAnomalyModelsRequest(ConfigRequest):
    drax_fusion_mode: str = "average"
    backbone_mode: str = "3d"
    backbone_temporal_kernel_size: int = 3


__all__ = [
    "BenchmarkVideoAnomalyRequest",
    "InferVideoAnomalyRequest",
    "ListVideoAnomalyModelsRequest",
    "TrainVideoAnomalyRequest",
    "VideoAnomalyRequest",
]
