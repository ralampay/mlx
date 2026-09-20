"""Video anomaly detection public workflows, loaded on demand."""
from importlib import import_module

_EXPORTS = {
    "BenchmarkVideoAnomalyModel": "evaluation",
    "BenchmarkVideoAnomalyRequest": "requests",
    "InferVideoAnomaly": "inference",
    "InferVideoAnomalyRequest": "requests",
    "VideoAnomalyInferenceResult": "inference",
    "ListVideoAnomalyModels": "list_models",
    "ListVideoAnomalyModelsRequest": "requests",
    "TrainVideoAnomalyModel": "training",
    "TrainVideoAnomalyRequest": "requests",
    "VideoAnomalyModel": "models",
    "VideoAnomaly3DModel": "models",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
