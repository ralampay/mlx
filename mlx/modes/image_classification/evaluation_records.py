from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OneShotPairRecord:
    image_one: Path
    label_one: str
    image_two: Path
    label_two: str
    target: int

@dataclass(frozen=True)
class OneShotPredictionRecord:
    image_one: Path
    label_one: str
    image_two: Path
    label_two: str
    target: int
    predicted: int
    same_probability: float
