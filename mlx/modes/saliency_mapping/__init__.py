"""Still-image salient-object-detection workflows."""

from mlx.modes.saliency_mapping.benchmark import BenchmarkSaliencyMapping
from mlx.modes.saliency_mapping.data import BuildSaliencyDataset
from mlx.modes.saliency_mapping.inference import InferSaliencyImage
from mlx.modes.saliency_mapping.train import (
    SmokeTestSaliencyModels,
    TrainSaliencyModel,
)

__all__ = [
    "BenchmarkSaliencyMapping",
    "BuildSaliencyDataset",
    "InferSaliencyImage",
    "SmokeTestSaliencyModels",
    "TrainSaliencyModel",
]
