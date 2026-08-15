from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, Sequence

from mlx.core.commands import WorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary
from mlx.modes.object_detection.models import DetectionAdapter
from mlx.modes.object_detection.requests import (
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    TrainObjectDetectionRequest,
)


class ObjectDetectionProvider(Protocol):
    name: str

    def train(self, request: TrainObjectDetectionRequest, reporter: WorkflowReporter) -> Any:
        ...

    def create_detector(self, request: ObjectDetectionRequest) -> DetectionAdapter:
        ...

    def convert(self, request: ConvertObjectDetectionRequest, reporter: WorkflowReporter):
        ...

    def list_models(
        self,
        request: ListObjectDetectionModelsRequest,
        reporter: WorkflowReporter,
    ) -> Sequence[ModelParameterSummary]:
        ...


PROVIDER_REGISTRY: dict[str, str] = {
    "libreyolo": "mlx.modes.object_detection.libreyolo.provider:get_provider",
    "ultralytics": "mlx.modes.object_detection.ultralytics.provider:get_provider",
}


def register_provider(name: str, factory_path: str) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Provider name cannot be empty.")
    PROVIDER_REGISTRY[normalized] = factory_path


def get_provider(name: str) -> ObjectDetectionProvider:
    normalized = (name or "ultralytics").strip().lower()
    factory_path = PROVIDER_REGISTRY.get(normalized)
    if factory_path is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise MLXUserError(
            f"Unsupported object-detection provider '{name}'. Available providers: {available}."
        )

    module_name, attribute_name = factory_path.split(":", 1)
    try:
        module = import_module(module_name)
        factory = getattr(module, attribute_name)
        return factory()
    except MLXUserError:
        raise
    except (ImportError, AttributeError) as exc:
        raise MLXUserError(
            f"Object-detection provider '{normalized}' is unavailable: {exc}. "
            "Install the provider dependencies and try again."
        ) from exc
