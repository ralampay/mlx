from __future__ import annotations

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary, count_model_parameters
from mlx.modes.object_detection.libreyolo.model_factory import build_scratch_model
from mlx.modes.object_detection.libreyolo.utils import (
    CANONICAL_MODEL_NAMES,
    MODEL_SPECS,
)


class ListLibreYOLOModels:
    def execute(self) -> list[ModelParameterSummary]:
        summaries = []
        for model_name in CANONICAL_MODEL_NAMES:
            try:
                model_spec = MODEL_SPECS[model_name]
                model = build_scratch_model(model_spec, device="cpu")
                summaries.append(
                    ModelParameterSummary(
                        model_name=model_name,
                        parameter_count=count_model_parameters(model.model),
                    )
                )
                del model
            except (AttributeError, ImportError, TypeError, ValueError, RuntimeError) as exc:
                raise MLXUserError(
                    f"Failed to construct LibreYOLO model '{model_name}' for listing: {exc}"
                ) from exc
        return summaries
