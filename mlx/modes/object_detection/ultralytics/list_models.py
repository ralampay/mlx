from __future__ import annotations

from mlx.core.model_listing import ModelParameterSummary, count_model_parameters
from mlx.modes.object_detection.ultralytics.utils import (
    initialize_model,
    resolve_model_paths,
)


CANONICAL_MODEL_NAMES = ("draxnet-yolo26", "yolo26")


class ListObjectDetectionModels:
    def execute(self) -> list[ModelParameterSummary]:
        summaries = []
        for model_name in CANONICAL_MODEL_NAMES:
            resolved_cfg, _ = resolve_model_paths(
                {"model": model_name},
                require_yaml=True,
                require_weights=False,
            )
            model = initialize_model(resolved_cfg, None, prefer_cfg=True)
            summaries.append(
                ModelParameterSummary(
                    model_name=model_name,
                    parameter_count=count_model_parameters(model.model),
                )
            )
            del model
        return summaries
