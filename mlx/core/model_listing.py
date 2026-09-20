from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ComponentSummary:
    name: str
    description: str = ""


class ListComponentNames:
    """Metadata-only discovery over an explicitly supplied inventory."""

    def __init__(self, names, *, descriptions=None):
        self.names = tuple(names)
        self.descriptions = dict(descriptions or {})

    def execute(self) -> tuple[ComponentSummary, ...]:
        return tuple(ComponentSummary(name, self.descriptions.get(name, "")) for name in sorted(self.names))


class ParameterizedModel(Protocol):
    def parameters(self):
        """Return an iterable of model parameters."""


@dataclass(frozen=True)
class ModelParameterSummary:
    model_name: str
    parameter_count: int


def count_model_parameters(model: ParameterizedModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
