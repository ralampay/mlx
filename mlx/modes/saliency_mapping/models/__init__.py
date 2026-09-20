from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from mlx.core.extensions import load_reference

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import count_model_parameters
from mlx.modes.saliency_mapping.compatibility import (
    segmentation_inventory, build_segmentation_saliency, segmentation_metadata,
)

DEFAULT_MODEL = "unet"
_names, SMALL_MODEL_NAMES = segmentation_inventory()
MODEL_NAMES = frozenset(_names)


@dataclass(frozen=True)
class SaliencyModelRegistry:
    """Builders return modules producing one-channel logits, not probabilities."""
    entries: Mapping[str, Callable | str] = field(default_factory=lambda: {
        name: build_segmentation_saliency for name in MODEL_NAMES
    })
    small_names: frozenset[str] = SMALL_MODEL_NAMES

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "small_names", frozenset(self.small_names))

    def register(self, name, builder, *, small=False):
        name = name.strip().lower()
        if not name:
            raise ValueError("Saliency model name cannot be empty.")
        groups = self.small_names | {name} if small else self.small_names
        return SaliencyModelRegistry({**self.entries, name: builder}, groups)

    def resolve(self, name):
        value = name if ":" in name else self.entries.get(name)
        if value is None:
            raise MLXUserError(f"Unsupported saliency model '{name}'.")
        builder = load_reference(value, kind="saliency model") if isinstance(value, str) else value
        if not callable(builder):
            raise MLXUserError(f"Saliency model '{name}' builder must be callable.")
        return builder


DEFAULT_SALIENCY_REGISTRY = SaliencyModelRegistry()

MODEL_GROUP_NAMES = frozenset({"all", "all-small"})


@dataclass(frozen=True)
class SaliencyModelSummary:
    model_name: str
    parameter_count: int
    backbone: str
    pretrained_supported: bool
    groups: tuple[str, ...]
    output_channels: int = 1


def supported_model_names(registry=None) -> list[str]:
    return sorted((registry or DEFAULT_SALIENCY_REGISTRY).entries)


def grouped_model_names(selector: str, *, registry=None) -> list[str]:
    if selector == "all":
        return supported_model_names(registry)
    if selector == "all-small":
        return sorted((registry or DEFAULT_SALIENCY_REGISTRY).small_names & set(supported_model_names(registry)))
    if ":" in selector or selector in supported_model_names(registry):
        return [selector]
    available = ", ".join((*sorted(MODEL_GROUP_NAMES), *supported_model_names(registry)))
    raise MLXUserError(
        f"Unsupported saliency model or group '{selector}'. Available values: {available}."
    )


def build_saliency_model(model_name: str, config: dict[str, Any], *, registry=None):
    builder = (registry or DEFAULT_SALIENCY_REGISTRY).resolve(model_name)
    try:
        return builder(model_name, config)
    except (TypeError, ValueError) as exc:
        raise MLXUserError(f"Cannot build saliency model '{model_name}': {exc}") from exc


def model_summary(model_name: str, config: dict[str, Any], *, registry=None) -> SaliencyModelSummary:
    listing_config = {**config, "pretrained": False}
    model = build_saliency_model(model_name, listing_config, registry=registry)
    groups = ["all"]
    if model_name in (registry or DEFAULT_SALIENCY_REGISTRY).small_names:
        groups.append("all-small")
    backbone, pretrained_supported = segmentation_metadata(model_name) if model_name in MODEL_NAMES else ("custom", False)
    summary = SaliencyModelSummary(
        model_name=model_name,
        parameter_count=count_model_parameters(model),
        backbone=backbone,
        pretrained_supported=pretrained_supported,
        groups=tuple(groups),
    )
    del model
    return summary


__all__ = [
    "SaliencyModelRegistry",
    "DEFAULT_SALIENCY_REGISTRY",
    "DEFAULT_MODEL",
    "MODEL_GROUP_NAMES",
    "MODEL_NAMES",
    "SMALL_MODEL_NAMES",
    "SaliencyModelSummary",
    "build_saliency_model",
    "grouped_model_names",
    "model_summary",
    "supported_model_names",
]
