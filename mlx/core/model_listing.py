from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ParameterizedModel(Protocol):
    def parameters(self):
        """Return an iterable of model parameters."""


@dataclass(frozen=True)
class ModelParameterSummary:
    model_name: str
    parameter_count: int


def count_model_parameters(model: ParameterizedModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
