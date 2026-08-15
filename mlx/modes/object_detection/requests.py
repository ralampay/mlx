from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ObjectDetectionRequest:
    provider: str = "ultralytics"
    model: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "cpu"
    height: int = 640
    width: int = 640
    confidence: float = 0.25

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        fields = cls.__dataclass_fields__
        values = {name: config[name] for name in fields if name in config}
        return cls(**values)

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainObjectDetectionRequest(ObjectDetectionRequest):
    dataset_path: str = "./tmp/dataset"
    output_path: Optional[str] = None
    epochs: int = 100
    batch_size: int = 16
    use_best: bool = True
    run_name: Optional[str] = None
    pretrained: bool = False
    optimizer: str = "auto"
    nbs: int = 64
    warmup_epochs: float = 3.0
    amp: bool = True
    lr0: Optional[float] = None
    loss_clip: Optional[float] = None
    random_seed: Optional[int] = None
    plots: bool = True


@dataclass(frozen=True)
class ConvertObjectDetectionRequest(ObjectDetectionRequest):
    output_path: Optional[str] = None


@dataclass(frozen=True)
class StreamObjectDetectionRequest(ObjectDetectionRequest):
    source: str = "camera"
    file_path: Optional[str] = None
    camera_index: int = 0


@dataclass(frozen=True)
class ListObjectDetectionModelsRequest:
    provider: str = "ultralytics"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        return cls(provider=str(config.get("provider") or "ultralytics"))

