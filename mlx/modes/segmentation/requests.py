from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class SegmentationRequest(ConfigRequest):
    action: str = "test"
    model: Optional[str] = None
    model_path: Optional[str] = None
    dataset_path: str = ""
    dataset_s3_uri: Optional[str] = None
    dataset_cache_dir: str = "~/.cache/mlx/datasets"
    output_path: Optional[str] = None
    input_img: str = "/tmp/image.jpg"
    file_path: Optional[str] = None
    device: str = "cpu"
    width: int = 256
    height: int = 256
    input_size: tuple[int, int] = (256, 256)
    batch_size: int = 4
    epochs: int = 50
    lr: Optional[float] = None
    colored: bool = True
    pretrained: bool = False
    num_classes: int = 2
    class_names: Optional[str] = None
    split: str = "test"
    camera_index: int = 0
    overlay_alpha: float = 0.45
    mask_threshold: float = 0.5
    display: bool = False
    random_seed: Optional[int] = None


@dataclass(frozen=True)
class TrainSegmentationRequest(SegmentationRequest):
    action: str = "train"


@dataclass(frozen=True)
class BenchmarkSegmentationRequest(SegmentationRequest):
    action: str = "benchmark"
    boundary_tolerance: int = 2
    calibration_bins: int = 15
    threshold_steps: int = 101
    plots: bool = True


@dataclass(frozen=True)
class InferSegmentationRequest(SegmentationRequest):
    action: str = "infer-image"


@dataclass(frozen=True)
class SmokeTestSegmentationRequest(SegmentationRequest):
    action: str = "test"


@dataclass(frozen=True)
class BuildSegmentationDatasetRequest(ConfigRequest):
    dataset_path: str = ""
    output_path: Optional[str] = None
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None
    overwrite: bool = False
    random_seed: Optional[int] = None
