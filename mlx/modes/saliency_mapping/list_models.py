from __future__ import annotations

from typing import Any

from mlx.modes.saliency_mapping.models import model_summary, supported_model_names


class ListSaliencyModels:
    def __init__(self, config: dict[str, Any], *, model_registry=None) -> None:
        self.model_registry = model_registry
        self.config = dict(config)

    def execute(self):
        return [model_summary(name, self.config, registry=self.model_registry) for name in supported_model_names(self.model_registry)]


__all__ = ["ListSaliencyModels"]
