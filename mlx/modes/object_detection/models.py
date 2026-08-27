from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class DetectionResult:
    detections: Sequence[Detection]
    names: Mapping[int, str]


@dataclass(frozen=True)
class ObjectDetectionBenchmarkResult:
    provider: str
    model_path: str
    dataset: str
    split: str
    metrics: Mapping[str, float]
    output_dir: Path
    evaluation_backend: str
    native_metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values


@dataclass(frozen=True)
class ObjectDetectionTrainingResult:
    training_result: Any
    checkpoint_path: str
    benchmark_result: ObjectDetectionBenchmarkResult


class DetectionAdapter(Protocol):
    def predict(self, frame: np.ndarray) -> DetectionResult:
        ...
