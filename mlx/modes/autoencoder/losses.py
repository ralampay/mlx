from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.models import _load_definition


@runtime_checkable
class ReconstructionLossDefinition(Protocol):
    name: str
    description: str

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        ...


class MSELossDefinition:
    name = "mse"
    description = "Mean squared reconstruction error."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        _reject_options(self.name, config)
        return nn.MSELoss()


class MAELossDefinition:
    name = "mae"
    description = "Mean absolute reconstruction error."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        _reject_options(self.name, config)
        return nn.L1Loss()


class SmoothL1LossDefinition:
    name = "smooth-l1"
    description = "Robust Smooth L1 reconstruction error; accepts positive beta."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        unknown = set(config) - {"beta"}
        if unknown:
            raise MLXUserError(f"Unsupported smooth-l1 option(s): {', '.join(sorted(unknown))}.")
        beta = float(config.get("beta", 1.0))
        if beta <= 0:
            raise MLXUserError("smooth-l1 beta must be greater than zero.")
        return nn.SmoothL1Loss(beta=beta)


def _reject_options(name: str, config: Mapping[str, Any]) -> None:
    if config:
        raise MLXUserError(f"Loss '{name}' does not accept configuration options.")


BUILTIN_LOSSES: Mapping[str, str] = MappingProxyType(
    {
        "mae": "mlx.modes.autoencoder.losses:MAELossDefinition",
        "mse": "mlx.modes.autoencoder.losses:MSELossDefinition",
        "smooth-l1": "mlx.modes.autoencoder.losses:SmoothL1LossDefinition",
    }
)


@dataclass(frozen=True)
class ReconstructionLossRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: BUILTIN_LOSSES)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entries",
            MappingProxyType({str(key).strip().lower(): str(value) for key, value in self.entries.items()}),
        )

    def register(self, name: str, definition_path: str) -> "ReconstructionLossRegistry":
        normalized = name.strip().lower()
        if not normalized or ":" not in definition_path:
            raise ValueError("Loss registration requires a name and package.module:DefinitionClass path.")
        return ReconstructionLossRegistry({**self.entries, normalized: definition_path})

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def resolve(self, reference: str) -> tuple[ReconstructionLossDefinition, str]:
        requested = reference.strip()
        path = requested if ":" in requested else self.entries.get(requested.lower())
        if path is None:
            available = ", ".join(self.names()) or "none"
            raise MLXUserError(
                f"Unsupported autoencoder loss '{reference}'. Available losses: {available}; "
                "external losses may use package.module:DefinitionClass."
            )
        definition = _load_definition(path, "loss")
        if not isinstance(definition, ReconstructionLossDefinition):
            raise MLXUserError(
                f"Loss definition '{path}' must provide name, description, and build(config)."
            )
        return definition, path


DEFAULT_LOSS_REGISTRY = ReconstructionLossRegistry()


__all__ = [
    "DEFAULT_LOSS_REGISTRY",
    "MAELossDefinition",
    "MSELossDefinition",
    "ReconstructionLossDefinition",
    "ReconstructionLossRegistry",
    "SmoothL1LossDefinition",
]
