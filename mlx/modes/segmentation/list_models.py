from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary, count_model_parameters
from mlx.modes.segmentation.models import (
    build_segmentation_model,
    supported_model_names,
)


class ListSegmentationModels:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def execute(self) -> list[ModelParameterSummary]:
        num_classes = int(self.config.get("num_classes", 2))
        if num_classes < 1:
            raise MLXUserError("--num-classes must be at least 1 when listing models.")

        summaries = []
        for model_name in supported_model_names():
            model = build_segmentation_model(
                model_name,
                self.config,
                num_classes=num_classes,
            )
            summaries.append(
                ModelParameterSummary(
                    model_name=model_name,
                    parameter_count=count_model_parameters(model),
                )
            )
            del model
        return summaries
