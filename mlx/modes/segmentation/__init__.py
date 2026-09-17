"""Segmentation mode package."""

from mlx.modes.segmentation.data import BuildSegmentationDataset
from mlx.modes.segmentation.evaluation import BenchmarkSegmentation
from mlx.modes.segmentation.inference import (
    InferSegmentationImage,
    RunSegmentationStreamInference,
)
from mlx.modes.segmentation.samples import GenerateSegmentationSamples
from mlx.modes.segmentation.train import (
    SmokeTestSegmentationModel,
    TrainSegmentationModel,
)
from mlx.modes.segmentation.train_all import (
    SegmentationAllModelsResult,
    SegmentationModelEvaluation,
    TrainAllSegmentationModels,
)

__all__ = [
    "BenchmarkSegmentation",
    "BuildSegmentationDataset",
    "GenerateSegmentationSamples",
    "InferSegmentationImage",
    "RunSegmentationStreamInference",
    "SegmentationAllModelsResult",
    "SegmentationModelEvaluation",
    "SmokeTestSegmentationModel",
    "TrainAllSegmentationModels",
    "TrainSegmentationModel",
]
