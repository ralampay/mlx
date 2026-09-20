"""Still-image salient-object-detection public commands, loaded on demand."""
from importlib import import_module

_EXPORTS = {
    "BenchmarkSaliencyMapping": "benchmark",
    "BuildSaliencyDataset": "data",
    "InferSaliencyImage": "inference",
    "SmokeTestSaliencyModels": "train",
    "TrainSaliencyModel": "train",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
