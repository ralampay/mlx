from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class SaliencyRequest(ConfigRequest):
    action: str = "test"
    model: Optional[str] = None
    model_path: Optional[str] = None
    dataset_path: str = ""
    dataset_s3_uri: Optional[str] = None
    dataset_cache_dir: str = "~/.cache/mlx/datasets"
    output_path: Optional[str] = None
    input_img: str = "/tmp/image.jpg"
    device: str = "cpu"
    width: int = 256
    height: int = 256
    input_size: tuple[int, int] = (256, 256)
    transform: str = "resize"
    batch_size: int = 4
    epochs: int = 50
    lr: Optional[float] = None
    colored: bool = True
    pretrained: bool = False
    split: str = "test"
    overlay_alpha: float = 0.45
    threshold_steps: int = 101
    mask_threshold: float = 0.5
    bce_weight: float = 1.0
    ssim_weight: float = 1.0
    iou_weight: float = 1.0
    workers: int = 0
    plots: bool = True
    save_images: bool = True
    random_seed: Optional[int] = None


@dataclass(frozen=True)
class TrainSaliencyRequest(SaliencyRequest):
    action: str = "train"


@dataclass(frozen=True)
class BenchmarkSaliencyRequest(SaliencyRequest):
    action: str = "benchmark"


@dataclass(frozen=True)
class BuildSaliencyDatasetRequest(ConfigRequest):
    dataset_path: str = ""
    output_path: Optional[str] = None
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None
    overwrite: bool = False
    random_seed: Optional[int] = None
