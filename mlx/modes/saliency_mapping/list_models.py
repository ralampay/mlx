from __future__ import annotations

from typing import Any

from mlx.modes.saliency_mapping.models import model_summary, supported_model_names


class ListSaliencyModels:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def execute(self):
        return [model_summary(name, self.config) for name in supported_model_names()]


__all__ = ["ListSaliencyModels"]
