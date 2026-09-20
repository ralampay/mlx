from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError


@runtime_checkable
class VectorAutoencoder(Protocol):
    input_dimensions: int
    bottleneck_dimensions: int

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class AutoencoderDefinition(Protocol):
    name: str
    description: str

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        ...


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dimensions: int, hidden_dimensions: int, bottleneck_dimensions: int) -> None:
        super().__init__()
        if input_dimensions < 2:
            raise ValueError("input_dimensions must be at least 2")
        if not 1 <= bottleneck_dimensions < input_dimensions:
            raise ValueError("bottleneck_dimensions must be positive and smaller than input_dimensions")
        if hidden_dimensions < bottleneck_dimensions:
            raise ValueError("hidden_dimensions must be at least bottleneck_dimensions")
        self.input_dimensions = int(input_dimensions)
        self.hidden_dimensions = int(hidden_dimensions)
        self.bottleneck_dimensions = int(bottleneck_dimensions)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dimensions, self.hidden_dimensions),
            nn.GELU(),
            nn.Linear(self.hidden_dimensions, self.bottleneck_dimensions),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dimensions, self.hidden_dimensions),
            nn.GELU(),
            nn.Linear(self.hidden_dimensions, self.input_dimensions),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder(embeddings)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


class SimpleAutoencoderDefinition:
    name = "simple"
    description = "Symmetric GELU MLP with a linear bottleneck and reconstruction output."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        try:
            return SimpleAutoencoder(
                input_dimensions=int(config["input_dimensions"]),
                hidden_dimensions=int(config.get("hidden_dimensions", 256)),
                bottleneck_dimensions=int(config.get("bottleneck_dimensions", 128)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MLXUserError(f"Invalid simple autoencoder configuration: {exc}") from exc


BUILTIN_AUTOENCODERS: Mapping[str, str] = MappingProxyType(
    {"simple": "mlx.modes.autoencoder.models:SimpleAutoencoderDefinition"}
)


@dataclass(frozen=True)
class AutoencoderRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: BUILTIN_AUTOENCODERS)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entries",
            MappingProxyType({str(key).strip().lower(): str(value) for key, value in self.entries.items()}),
        )

    def register(self, name: str, definition_path: str) -> "AutoencoderRegistry":
        normalized = name.strip().lower()
        if not normalized or ":" not in definition_path:
            raise ValueError("Autoencoder registration requires a name and package.module:DefinitionClass path.")
        return AutoencoderRegistry({**self.entries, normalized: definition_path})

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def resolve(self, reference: str) -> tuple[AutoencoderDefinition, str]:
        requested = reference.strip()
        path = requested if ":" in requested else self.entries.get(requested.lower())
        if path is None:
            available = ", ".join(self.names()) or "none"
            raise MLXUserError(
                f"Unsupported autoencoder model '{reference}'. Available models: {available}; "
                "external models may use package.module:DefinitionClass."
            )
        definition = _load_definition(path, "autoencoder")
        if not isinstance(definition, AutoencoderDefinition):
            raise MLXUserError(
                f"Autoencoder definition '{path}' must provide name, description, and build(config)."
            )
        return definition, path


def _load_definition(path: str, kind: str):
    if ":" not in path:
        raise MLXUserError(f"{kind.title()} import path must use package.module:ClassName format.")
    module_name, attribute = path.split(":", 1)
    try:
        value = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError, ValueError) as exc:
        raise MLXUserError(f"Unable to load {kind} definition '{path}': {exc}") from exc
    if not inspect.isclass(value):
        raise MLXUserError(f"{kind.title()} import '{path}' does not reference a class.")
    try:
        return value()
    except (TypeError, ValueError) as exc:
        raise MLXUserError(f"Unable to construct {kind} definition '{path}': {exc}") from exc


DEFAULT_AUTOENCODER_REGISTRY = AutoencoderRegistry()


__all__ = [
    "AutoencoderDefinition",
    "AutoencoderRegistry",
    "DEFAULT_AUTOENCODER_REGISTRY",
    "SimpleAutoencoder",
    "SimpleAutoencoderDefinition",
    "VectorAutoencoder",
]
