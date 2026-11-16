"""Immutable, lazy model selection; explicit caller registries establish trust."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference

BUILTINS = MappingProxyType(
    {
        "vanilla-vae": "mlx.modes.time_series.models:VanillaDefinition",
        "conv-vae": "mlx.modes.time_series.models:ConvDefinition",
        "timevae": "mlx.modes.time_series.models:TimeVAEDefinition",
    }
)


@dataclass(frozen=True)
class TemporalModelRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: BUILTINS)

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def resolve(self, name):
        if name not in self.entries:
            raise MLXUserError(
                f"Unknown temporal model '{name}'. Choose from {', '.join(self.entries)}."
            )
        definition = load_reference(self.entries[name], kind="temporal model")()
        if not callable(getattr(definition, "build", None)):
            raise MLXUserError(
                "Temporal model definitions must implement build(config)."
            )
        return definition

    def inventory(self):
        return [
            {"name": name, "description": self.resolve(name).description}
            for name in self.entries
        ]
