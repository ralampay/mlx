from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    names: dict[int, str]


class DetectionAdapter(Protocol):
    def predict(self, frame: np.ndarray) -> DetectionResult:
        ...
