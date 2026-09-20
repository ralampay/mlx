"""Segmentation public commands, loaded only when requested."""
from importlib import import_module

_EXPORTS = {
    "BenchmarkSegmentation": "evaluation",
    "BuildSegmentationDataset": "data",
    "GenerateSegmentationSamples": "samples",
    "InferSegmentationImage": "inference",
    "RunSegmentationStreamInference": "inference",
    "SegmentationAllModelsResult": "train_all",
    "SegmentationModelEvaluation": "train_all",
    "SmokeTestSegmentationModel": "train",
    "TrainAllSegmentationModels": "train_all",
    "TrainSegmentationModel": "train",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
