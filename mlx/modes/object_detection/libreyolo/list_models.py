from __future__ import annotations

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary, count_model_parameters
from mlx.modes.object_detection.libreyolo.utils import (
    CANONICAL_MODEL_NAMES,
    MODEL_SPECS,
    build_drax_config,
    dependency_error,
)


class ListLibreYOLOModels:
    def execute(self) -> list[ModelParameterSummary]:
        try:
            from libreyolo import LibreYOLO9
        except ImportError as exc:
            raise dependency_error("listing object-detection models") from exc

        summaries = []
        for model_name in CANONICAL_MODEL_NAMES:
            try:
                model_spec = MODEL_SPECS[model_name]
                model_kwargs = {
                    "model_path": None,
                    "size": model_spec.size,
                    "device": "cpu",
                    "task": "detect",
                }
                if model_spec.uses_drax:
                    model_kwargs["drax_config"] = build_drax_config(model_spec)
                model = LibreYOLO9(**model_kwargs)
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
