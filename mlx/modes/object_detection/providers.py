from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from mlx.core.commands import WorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary
from mlx.modes.object_detection.models import DetectionAdapter
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    TrainObjectDetectionRequest,
)


class ObjectDetectionProvider(Protocol):
    name: str

    def train(self, request: TrainObjectDetectionRequest, reporter: WorkflowReporter) -> Any:
        ...

    def benchmark(
        self,
        request: BenchmarkObjectDetectionRequest,
        reporter: WorkflowReporter,
    ) -> Any:
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


@dataclass(frozen=True)
class ProviderRegistry:
    entries: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def register(self, name: str, factory_path: str) -> "ProviderRegistry":
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Provider name cannot be empty.")
        if ":" not in factory_path:
            raise ValueError("Provider factory path must use 'package.module:function' format.")
        entries = dict(self.entries)
        entries[normalized] = factory_path
        return ProviderRegistry(entries)


_COMPAT_PROVIDER_ENTRIES = {
    "libreyolo": "mlx.modes.object_detection.libreyolo.provider:get_provider",
    "ultralytics": "mlx.modes.object_detection.ultralytics.provider:get_provider",
}
PROVIDER_REGISTRY = MappingProxyType(_COMPAT_PROVIDER_ENTRIES)
DEFAULT_PROVIDER_REGISTRY = ProviderRegistry(_COMPAT_PROVIDER_ENTRIES)


def register_provider(
    name: str,
    factory_path: str,
    *,
    registry: ProviderRegistry | None = None,
) -> ProviderRegistry:
    """Return an extended registry without mutating process-wide provider state."""

    global DEFAULT_PROVIDER_REGISTRY
    updated = (registry or DEFAULT_PROVIDER_REGISTRY).register(name, factory_path)
    if registry is None:
        _COMPAT_PROVIDER_ENTRIES[name.strip().lower()] = factory_path
        DEFAULT_PROVIDER_REGISTRY = ProviderRegistry(_COMPAT_PROVIDER_ENTRIES)
    return updated


def get_provider(
    name: str,
    *,
    registry: ProviderRegistry | None = None,
) -> ObjectDetectionProvider:
    normalized = (name or "ultralytics").strip().lower()
    selected = registry or DEFAULT_PROVIDER_REGISTRY
    factory_path = selected.entries.get(normalized)
    if factory_path is None:
        available = ", ".join(sorted(selected.entries))
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
