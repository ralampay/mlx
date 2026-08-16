from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking.protocols import TrackingAlgorithm


BUILTIN_TRACKERS: Mapping[str, str] = MappingProxyType(
    {
        "bytetrack": (
            "mlx.modes.object_detection.tracking.algorithms.bytetrack:ByteTrackAlgorithm"
        ),
        "sort": (
            "mlx.modes.object_detection.tracking.algorithms.sort:SortTrackingAlgorithm"
        ),
    }
)


def _validate_registration(name: str, class_path: str) -> tuple[str, str]:
    normalized = name.strip().lower()
    validated_path = class_path.strip()
    if not normalized:
        raise ValueError("Tracker name cannot be empty.")
    if ":" not in validated_path:
        raise ValueError("Tracker class path must use 'package.module:ClassName' format.")
    module_name, attribute_name = validated_path.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("Tracker class path must use 'package.module:ClassName' format.")
    return normalized, validated_path


@dataclass(frozen=True)
class TrackerRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: BUILTIN_TRACKERS)

    def __post_init__(self) -> None:
        normalized_entries = {}
        for name, class_path in self.entries.items():
            normalized, validated_path = _validate_registration(name, class_path)
            normalized_entries[normalized] = validated_path
        object.__setattr__(self, "entries", MappingProxyType(normalized_entries))

    def register(self, name: str, class_path: str) -> TrackerRegistry:
        normalized, validated_path = _validate_registration(name, class_path)
        entries = dict(self.entries)
        entries[normalized] = validated_path
        return TrackerRegistry(entries)

    def resolve(self, name: str) -> str | None:
        return self.entries.get(name.strip().lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))


DEFAULT_TRACKER_REGISTRY = TrackerRegistry()


def register_tracker(
    name: str,
    class_path: str,
    *,
    registry: TrackerRegistry | None = None,
) -> TrackerRegistry:
    """Return a new registry containing the requested tracker alias."""

    return (registry or DEFAULT_TRACKER_REGISTRY).register(name, class_path)


def list_trackers(*, registry: TrackerRegistry | None = None) -> tuple[str, ...]:
    return (registry or DEFAULT_TRACKER_REGISTRY).names()


class CreateTrackingAlgorithm:
    def __init__(
        self,
        *,
        tracker: str,
        config_path: str | None = None,
        options: dict[str, Any] | None = None,
        registry: TrackerRegistry | None = None,
    ) -> None:
        self.tracker = tracker
        self.config_path = config_path
        self.options = dict(options) if options is not None else None
        self.registry = registry or DEFAULT_TRACKER_REGISTRY

    def execute(self) -> TrackingAlgorithm:
        class_path = self._resolve_class_path()
        tracker_class = self._load_class(class_path)
        options = self._load_options()
        try:
            algorithm = tracker_class(**options)
        except ImportError as exc:
            raise MLXUserError(
                f"Tracker '{self.tracker}' is missing a required dependency: {exc}. "
                "Install the tracker package requirements and try again."
            ) from exc
        except TypeError as exc:
            raise MLXUserError(
                f"Tracker '{self.tracker}' rejected its configuration: {exc}. "
                "Inspect the tracker constructor and --tracker-config keys."
            ) from exc
        except ValueError as exc:
            raise MLXUserError(
                f"Invalid configuration for tracker '{self.tracker}': {exc}"
            ) from exc
        if not isinstance(algorithm, TrackingAlgorithm):
            raise MLXUserError(
                f"Tracker '{self.tracker}' must implement update(...) and reset()."
            )
        return algorithm

    def _resolve_class_path(self) -> str:
        normalized = (self.tracker or "bytetrack").strip()
        if ":" in normalized:
            return normalized
        class_path = self.registry.resolve(normalized)
        if class_path is None:
            available = ", ".join(self.registry.names())
            raise MLXUserError(
                f"Unsupported tracker '{self.tracker}'. Available trackers: {available}; "
                "external trackers may use package.module:ClassName."
            )
        return class_path

    def _load_class(self, class_path: str):
        module_name, attribute_name = class_path.split(":", 1)
        if not module_name or not attribute_name:
            raise MLXUserError(
                "Tracker class path must use 'package.module:ClassName' format."
            )
        try:
            value = getattr(import_module(module_name), attribute_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise MLXUserError(
                f"Unable to load tracker '{class_path}': {exc}. "
                "Check the import path and installed package."
            ) from exc
        if not inspect.isclass(value):
            raise MLXUserError(f"Tracker import '{class_path}' does not reference a class.")
        return value

    def _load_options(self) -> dict[str, Any]:
        if self.options is not None and self.config_path is not None:
            raise MLXUserError(
                "Tracker configuration must come from either injected options or "
                "--tracker-config, not both."
            )
        if self.options is not None:
            return dict(self.options)
        if self.config_path is None:
            return {}
        path = Path(self.config_path).expanduser()
        if not path.is_file():
            raise MLXUserError(f"Tracker configuration file not found: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MLXUserError(
                f"Unable to read tracker configuration '{path}': {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise MLXUserError("Tracker configuration must be a JSON object.")
        return value
