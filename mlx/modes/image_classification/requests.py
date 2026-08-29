from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class ImageClassificationRequest(ConfigRequest):
    action: str = "test"
    model: Optional[str] = None
    model_path: Optional[str] = None
    dataset_path: str = ""
    dataset_s3_uri: Optional[str] = None
    dataset_cache_dir: str = "~/.cache/mlx/datasets"
    output_path: Optional[str] = None
    input_img: str = "/tmp/image.jpg"
    device: str = "cpu"
    width: int = 224
    height: int = 224
    input_size: tuple[int, int] = (224, 224)
    batch_size: int = 1
    epochs: int = 100
    lr: Optional[float] = None
    colored: bool = True
    pretrained: bool = False
    embedding_size: int = 4096
    num_pairs: int = 100
    num_classes: int = 2
    random_seed: Optional[int] = None
    use_best: bool = True
    verbose: bool = False
    apply_transformations: bool = False
    display: bool = False
    ood_method: str = "none"
    svdd_weight: float = 0.05
    svdd_dim: int = 128
    svdd_hidden_dim: int = 256
    svdd_quantile: float = 0.95
    svdd_warmup_epochs: int = 0


@dataclass(frozen=True)
class TrainImageClassificationRequest(ImageClassificationRequest):
    action: str = "train"


@dataclass(frozen=True)
class BenchmarkImageClassificationRequest(ImageClassificationRequest):
    action: str = "benchmark"
    plots: bool = True


@dataclass(frozen=True)
class InferImageClassificationRequest(ImageClassificationRequest):
    action: str = "infer-image"


@dataclass(frozen=True)
class GenerateImageClassificationCamsRequest(ImageClassificationRequest):
    action: str = "cam"
    cam_method: str = "gradcam"
    target_layer: Optional[str] = None
    target_index: Optional[int] = None
    max_samples: Optional[int] = None
    save_images: bool = True
    window_delay: int = 0
    aug_smooth: bool = False
    eigen_smooth: bool = False


@dataclass(frozen=True)
class SmokeTestImageClassificationRequest(ImageClassificationRequest):
    action: str = "test"


@dataclass(frozen=True)
class BuildImageClassificationDatasetRequest(ConfigRequest):
    dataset_path: str = ""
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None
    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    split_mode: Optional[str] = None
    output_path: Optional[str] = None
    overwrite: bool = False
    random_seed: Optional[int] = None
