from __future__ import annotations

import gc
from dataclasses import dataclass

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.model_listing import count_model_parameters
from mlx.modes.video_anomaly_detection.models.backbone3d import (
    build_spatiotemporal_backbone_3d,
)
from mlx.modes.video_anomaly_detection.requests import ListVideoAnomalyModelsRequest
from mlx.modes.video_anomaly_detection.variants import video_anomaly_model_variants


@dataclass(frozen=True)
class VideoAnomalyBackboneSummary:
    model_name: str
    model_family: str
    feature_dim: int
    parameter_count: int
    pretrained_available: bool
    backbone_mode: str = "3d"
    backbone_class: str = ""
    drax_fusion_mode: str | None = None
    pretrained_provenance: str = "inflated_full"
    compatible: bool = True


class ListVideoAnomalyModels:
    def __init__(
        self,
        request: ListVideoAnomalyModelsRequest,
        *,
        reporter: WorkflowReporter | None = None,
        backbone_factory=build_spatiotemporal_backbone_3d,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.backbone_factory = backbone_factory

    def execute(self) -> tuple[VideoAnomalyBackboneSummary, ...]:
        config = {**self.request.to_config(), "colored": True, "pretrained": False}
        summaries = []
        for variant in video_anomaly_model_variants():
            name = variant.model_name
            fusion_mode = variant.drax_fusion_mode
            model_config = dict(config)
            if fusion_mode is not None:
                model_config["drax_fusion_mode"] = fusion_mode
            backbone = self.backbone_factory(name, model_config)
            summaries.append(
                VideoAnomalyBackboneSummary(
                    model_name=name,
                    model_family="standard",
                    feature_dim=int(backbone.feature_dim),
                    parameter_count=count_model_parameters(backbone),
                    pretrained_available=True,
                    backbone_mode="3d",
                    backbone_class=type(backbone).__name__,
                    drax_fusion_mode=fusion_mode,
                    pretrained_provenance=(
                        "inflated_partial" if name.startswith("drax") else "inflated_full"
                    ),
                )
            )
            del backbone
            gc.collect()
        emit(
            self.reporter,
            "success",
            f"Found {len(summaries)} compatible clip-native 3D backbone configurations.",
            payload={"event": "video_anomaly_models", "models": summaries},
        )
        return tuple(summaries)


__all__ = ["ListVideoAnomalyModels", "VideoAnomalyBackboneSummary"]
