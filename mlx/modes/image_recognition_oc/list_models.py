from __future__ import annotations

import gc
from dataclasses import dataclass

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.model_listing import count_model_parameters
from mlx.modes.image_recognition_oc.algorithms import (
    DEFAULT_ALGORITHM_REGISTRY,
    OneClassAlgorithmRegistry,
)
from mlx.modes.image_recognition_oc.backbones import image_backbone_names
from mlx.modes.image_recognition_oc.backbones import backbone_class_name
from mlx.modes.image_recognition_oc.requests import ListImageOneClassModelsRequest


@dataclass(frozen=True)
class ImageOneClassModelSummary:
    model_name: str
    backbone_name: str
    backbone_class: str
    feature_dim: int
    parameter_count: int
    pretrained_available: bool
    drax_fusion_mode: str | None = None


class ListImageOneClassModels:
    def __init__(
        self,
        request: ListImageOneClassModelsRequest,
        *,
        reporter: WorkflowReporter | None = None,
        registry: OneClassAlgorithmRegistry = DEFAULT_ALGORITHM_REGISTRY,
        backbone_names=image_backbone_names,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.registry = registry
        self.backbone_names = backbone_names

    def execute(self) -> tuple[ImageOneClassModelSummary, ...]:
        config = {**self.request.to_config(), "pretrained": False}
        selected_model = config.get("model")
        selected_backbone = config.get("backbone")
        algorithms = (
            (str(selected_model), self.registry.get(str(selected_model))),
        ) if selected_model else tuple(sorted(self.registry.algorithms.items()))
        backbones = (str(selected_backbone),) if selected_backbone else self.backbone_names()
        summaries = []
        for model_name, algorithm in algorithms:
            for backbone_name in backbones:
                model = algorithm.build_model(backbone_name, config)
                summaries.append(
                    ImageOneClassModelSummary(
                        model_name=model_name,
                        backbone_name=backbone_name,
                        backbone_class=backbone_class_name(model.backbone),
                        feature_dim=int(model.backbone_feature_dim),
                        parameter_count=count_model_parameters(model),
                        pretrained_available=True,
                        drax_fusion_mode=(
                            str(config.get("drax_fusion_mode", "average"))
                            if backbone_name.startswith("drax")
                            else None
                        ),
                    )
                )
                del model
                gc.collect()
        emit(
            self.reporter,
            "success",
            f"Found {len(summaries)} compatible one-class model/backbone configurations.",
            payload={"event": "image_one_class_models", "models": summaries},
        )
        return tuple(summaries)


__all__ = ["ImageOneClassModelSummary", "ListImageOneClassModels"]
