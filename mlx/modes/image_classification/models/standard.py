from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import Mapping

from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference
from mlx.modes.image_classification.models.catalog import TORCHVISION_MODELS, BUILTIN_BUILDERS


@dataclass(frozen=True)
class StandardModelRegistry:
    builders: Mapping[str, Callable | str] = field(default_factory=dict)
    feature_adapters: Mapping[str, Callable | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "builders", MappingProxyType(dict(self.builders)))
        object.__setattr__(self, "feature_adapters", MappingProxyType(dict(self.feature_adapters)))

    def register(self, name: str, builder: Callable | str, *, feature_adapter=None) -> "StandardModelRegistry":
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Standard model name cannot be empty.")
        builders = dict(self.builders)
        builders[normalized] = builder
        adapters = dict(self.feature_adapters)
        if feature_adapter is not None:
            adapters[normalized] = feature_adapter
        return StandardModelRegistry(builders, adapters)


_COMPAT_STANDARD_MODEL_BUILDERS: dict[str, Callable | str] = dict(BUILTIN_BUILDERS)
CUSTOM_STANDARD_MODEL_BUILDERS = MappingProxyType(_COMPAT_STANDARD_MODEL_BUILDERS)
DEFAULT_STANDARD_MODEL_REGISTRY = StandardModelRegistry(_COMPAT_STANDARD_MODEL_BUILDERS)

SUPPORTED_TORCHVISION_MODELS = frozenset(TORCHVISION_MODELS)


def register_standard_model(
    name: str,
    builder: Callable,
    *,
    registry: StandardModelRegistry | None = None,
) -> StandardModelRegistry:
    """Register against an injected registry, retaining legacy default registration."""

    global DEFAULT_STANDARD_MODEL_REGISTRY
    selected = registry or DEFAULT_STANDARD_MODEL_REGISTRY
    updated = selected.register(name, builder)
    if registry is None:
        _COMPAT_STANDARD_MODEL_BUILDERS[name.strip().lower()] = builder
        DEFAULT_STANDARD_MODEL_REGISTRY = StandardModelRegistry(
            _COMPAT_STANDARD_MODEL_BUILDERS
        )
    return updated


def registered_standard_model_names(
    registry: StandardModelRegistry | None = None,
) -> list[str]:
    return sorted((registry or DEFAULT_STANDARD_MODEL_REGISTRY).builders)


def build_standard_model(
    model_name: str,
    *,
    num_classes: int,
    colored: bool,
    pretrained: bool,
    config: dict | None = None,
    registry: StandardModelRegistry | None = None,
):
    custom_builder = model_name if ":" in model_name else (registry or DEFAULT_STANDARD_MODEL_REGISTRY).builders.get(model_name)
    if custom_builder is not None:
        if isinstance(custom_builder, str):
            custom_builder = load_reference(custom_builder, kind="classification model")
        builder_params = {
            "num_classes": num_classes,
            "colored": colored,
            "pretrained": pretrained,
        }
        if "config" in inspect.signature(custom_builder).parameters:
            builder_params["config"] = config or {}
        return custom_builder(**builder_params)

    if model_name in SUPPORTED_TORCHVISION_MODELS:
        try:
            from torchvision import models as torchvision_models
        except ImportError as exc:
            raise MLXUserError(
                "Torchvision is required for standard classification models. Install it with 'pip install torchvision'."
            ) from exc

        model, stem_attr = _build_torchvision_model(
            model_name=model_name,
            torchvision_models=torchvision_models,
            pretrained=pretrained,
        )

        if not colored:
            _replace_stem_conv(model, stem_attr)

        _replace_classifier_head(model, model_name, num_classes)
        return model

    raise MLXUserError(f"Unsupported standard image-classification model '{model_name}'.")


def _build_torchvision_model(*, model_name: str, torchvision_models, pretrained: bool):
    spec = TORCHVISION_MODELS[model_name]
    weights = getattr(torchvision_models, spec.weights).DEFAULT if pretrained else None
    return getattr(torchvision_models, spec.constructor)(weights=weights), spec.stem


def _replace_stem_conv(model, stem_attr: str) -> None:
    import torch

    original_conv = _resolve_module_attr(model, stem_attr)
    replacement_conv = nn.Conv2d(
        1,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    with torch.no_grad():
        replacement_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        if original_conv.bias is not None and replacement_conv.bias is not None:
            replacement_conv.bias.copy_(original_conv.bias)
    _assign_module_attr(model, stem_attr, replacement_conv)


def _replace_classifier_head(model, model_name: str, num_classes: int) -> None:
    head_path = TORCHVISION_MODELS[model_name].head
    head = _resolve_module_attr(model, head_path)
    _assign_module_attr(model, head_path, nn.Linear(head.in_features, num_classes))


def _resolve_module_attr(model, attr_path: str):
    module = model
    for part in attr_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _assign_module_attr(model, attr_path: str, new_module) -> None:
    parent_path, _, leaf = attr_path.rpartition(".")
    parent = _resolve_module_attr(model, parent_path) if parent_path else model
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)
