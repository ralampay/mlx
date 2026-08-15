"""Segmentation mode package."""

from mlx.modes.segmentation.data import BuildSegmentationDataset
from mlx.modes.segmentation.evaluation import BenchmarkSegmentation
from mlx.modes.segmentation.inference import (
    InferSegmentationImage,
    RunSegmentationStreamInference,
)
from mlx.modes.segmentation.train import (
    SmokeTestSegmentationModel,
    TrainSegmentationModel,
)

__all__ = [
    "BenchmarkSegmentation",
    "BuildSegmentationDataset",
    "InferSegmentationImage",
    "RunSegmentationStreamInference",
    "SmokeTestSegmentationModel",
    "TrainSegmentationModel",
]
