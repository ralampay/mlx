from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

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


class DetectionAdapter(Protocol):
    def predict(self, frame: np.ndarray) -> DetectionResult:
        ...
