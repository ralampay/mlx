"""Immutable definitions for MLX-owned feature adapters."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

from torch import nn

from mlx.core.extensions import load_reference, validate_reference


@dataclass(frozen=True)
class FeatureAdapterDefinition:
    """Construction and verification policy, independent of detector placement.

    Factories receive either a channel count or the base convolution, followed
    by the declared parameters. They must not modify base weights or structure.
    """
    factory: str | Callable
    attachment: str = "feature"
    parameters: tuple[str, ...] = ("reduction", "alpha")
    validate_target: Callable | None = None
    seed_initialization: bool = False
    verify_gradients: bool = False
    verify_recorded_targets: bool = False

    def __post_init__(self):
        if self.attachment not in {"feature", "conv"}:
            raise ValueError("Adapter attachment must be feature or conv")
        if isinstance(self.factory, str):
            validate_reference(self.factory)
        elif not callable(self.factory):
            raise ValueError("Adapter factory must be callable or an import reference")
        if self.validate_target is not None and not callable(self.validate_target):
            raise ValueError("Adapter target validator must be callable")
        if set(self.parameters) - {"reduction", "rank", "alpha"}:
            raise ValueError("Unknown adapter construction parameter")
        object.__setattr__(self, "parameters", tuple(self.parameters))

    def validate(self, module, channels):
        if self.attachment == "conv":
            if not isinstance(module, nn.Conv2d) or module.kernel_size != (1, 1) or module.groups != 1:
                raise ValueError("Adapter target must be a dense 1x1 Conv2d")
        elif channels < 1:
            raise ValueError("Feature adapter target needs a positive channel count")
        if self.validate_target is not None:
            self.validate_target(module, channels)

    def build(self, target, **options):
        factory = load_reference(self.factory, kind="feature adapter") if isinstance(self.factory, str) else self.factory
        result = factory(target, **{key: options[key] for key in self.parameters})
        if not isinstance(result, nn.Module):
            raise ValueError("Adapter factory must return a torch module")
        return result


def _unit_stride(module, channels):
    if module.stride != (1, 1) or module.padding != (0, 0):
        raise ValueError("Adapter convolution must have stride one and no padding")


@dataclass(frozen=True)
class FeatureAdapterRegistry:
    entries: Mapping[str, FeatureAdapterDefinition] = field(default_factory=dict)

    def __post_init__(self):
        entries = dict(self.entries)
        for name, definition in entries.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)
                    or name in {"frozen", "head-only", "full-finetune"}):
                raise ValueError("Adapter names must be lowercase slugs distinct from training strategies")
            if not isinstance(definition, FeatureAdapterDefinition):
                raise ValueError("Registry entries must be feature adapter definitions")
        object.__setattr__(self, "entries", MappingProxyType(entries))

    def register(self, name, definition):
        return FeatureAdapterRegistry({**self.entries, name: definition})

    def names(self):
        return tuple(self.entries)

    def resolve(self, name):
        try:
            return self.entries[name]
        except KeyError as exc:
            raise ValueError(f"Unknown feature adapter {name!r}; choose from {self.names()}") from exc


_PREFIX = "mlx.modes.object_detection.feature_adapters."
DEFAULT_FEATURE_ADAPTER_REGISTRY = FeatureAdapterRegistry({
    **{name: FeatureAdapterDefinition(_PREFIX + "layers:" + cls, parameters=params)
       for name, cls, params in (
           ("bottleneck", "BottleneckAdapter", ("reduction", "alpha")),
           ("ssf", "SSFAdapter", ()),
           ("convpass", "ConvpassAdapter", ("reduction", "alpha")),
           ("conv-adapter", "ConvAdapter", ("reduction", "alpha")),
           ("drax", "DraxAdapter", ("reduction", "alpha")),
       )},
    "lora": FeatureAdapterDefinition(_PREFIX + "layers:LoRAConv2d", "conv", ("rank", "alpha")),
    "drax-hybrid": FeatureAdapterDefinition(_PREFIX + "hybrid:DraxHybridConv2d", "conv",
        ("rank", "reduction", "alpha"), _unit_stride, seed_initialization=True,
        verify_gradients=True, verify_recorded_targets=True),
    "drax-spatial": FeatureAdapterDefinition(_PREFIX + "hybrid:DraxSpatialConv2d", "conv",
        ("reduction", "alpha"), _unit_stride, verify_recorded_targets=True),
    "drax-residual-fusion": FeatureAdapterDefinition(_PREFIX + "residual_fusion:DraxResidualFusionConv2d", "conv",
        ("reduction", "alpha"), _unit_stride, seed_initialization=True,
        verify_gradients=True, verify_recorded_targets=True),
})
