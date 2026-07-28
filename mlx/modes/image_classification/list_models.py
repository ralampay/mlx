from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary, count_model_parameters
from mlx.modes.image_classification.models import (
    build_image_classification_model,
    model_family_for,
    supported_model_names,
)


class ListImageClassificationModels:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = {**config, "pretrained": False}

    def execute(self) -> list[ModelParameterSummary]:
        num_classes = int(self.config.get("num_classes", 2))
        if num_classes < 1:
            raise MLXUserError("--num-classes must be at least 1 when listing models.")

        summaries = []
        for model_name in supported_model_names():
            model = self._build_model(model_name, num_classes)
            summaries.append(
                ModelParameterSummary(
                    model_name=model_name,
                    parameter_count=count_model_parameters(model),
                )
            )
            del model
        return summaries

    def _build_model(self, model_name: str, num_classes: int):
        if model_family_for(model_name) == "one-shot":
            return build_image_classification_model(model_name, self.config)
        return build_image_classification_model(
            model_name,
            self.config,
            num_classes=num_classes,
        )
