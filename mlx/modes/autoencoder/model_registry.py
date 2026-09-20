from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.definitions import load_definition
from mlx.modes.autoencoder.contracts import AutoencoderDefinition

BUILTIN_AUTOENCODERS: Mapping[str, str] = MappingProxyType(
    {
        "simple": "mlx.modes.autoencoder.models:SimpleAutoencoderDefinition",
        "tiny": "mlx.modes.autoencoder.architectures.tiny:TinyAutoencoderDefinition",
    }
)


@dataclass(frozen=True)
class AutoencoderRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: BUILTIN_AUTOENCODERS)
    descriptions: Mapping[str, str] = field(default_factory=lambda: {
        "simple": "Symmetric GELU MLP with a linear bottleneck and reconstruction output.",
        "tiny": "Two linear layers demonstrating the vector autoencoder contract.",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptions", MappingProxyType(dict(self.descriptions)))
        object.__setattr__(
            self,
            "entries",
            MappingProxyType({str(key).strip().lower(): str(value) for key, value in self.entries.items()}),
        )

    def register(self, name: str, definition_path: str, *, description: str = "") -> "AutoencoderRegistry":
        normalized = name.strip().lower()
        if not normalized or ":" not in definition_path:
            raise ValueError("Autoencoder registration requires a name and package.module:DefinitionClass path.")
        return AutoencoderRegistry({**self.entries, normalized: definition_path}, {**self.descriptions, normalized: description})

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
        definition = load_definition(path, "autoencoder")
        if not isinstance(definition, AutoencoderDefinition):
            raise MLXUserError(
                f"Autoencoder definition '{path}' must provide name, description, and build(config)."
            )
        return definition, path


DEFAULT_AUTOENCODER_REGISTRY = AutoencoderRegistry()
