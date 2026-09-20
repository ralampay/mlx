"""Classification public workflows, loaded on demand."""
from importlib import import_module

_EXPORTS = {
    "BenchmarkImageClassification": "evaluation",
    "BuildImageClassificationDataset": "data",
    "GenerateImageClassificationCams": "cam",
    "InferImageClassification": "inference",
    "SmokeTestImageClassificationModel": "train",
    "TrainImageClassificationModel": "train",
    "run_image_classification": "runner",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
