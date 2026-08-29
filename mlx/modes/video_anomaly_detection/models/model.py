from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.core.image_backbones import ImageFeatureBackboneFactory
from mlx.modes.video_anomaly_detection.models.backbone import (
    FrameBackbone,
    build_default_frame_backbone,
)
from mlx.modes.video_anomaly_detection.models.backbone3d import (
    InflatedImageBackbone3D,
    build_spatiotemporal_backbone_3d,
)
from mlx.modes.video_anomaly_detection.models.svdd import DeepSVDDHead
from mlx.modes.video_anomaly_detection.models.temporal import build_temporal_encoder


@dataclass(frozen=True)
class VideoAnomalyOutput:
    frame_features: torch.Tensor | None
    clip_embedding: torch.Tensor
    svdd_embedding: torch.Tensor
    anomaly_score: torch.Tensor


class VideoAnomalyModel(nn.Module):
    def __init__(
        self,
        frame_backbone: FrameBackbone,
        temporal_encoder: nn.Module,
        svdd_head: DeepSVDDHead,
    ) -> None:
        super().__init__()
        self.frame_backbone = frame_backbone
        self.temporal_encoder = temporal_encoder
        self.svdd_head = svdd_head

    @property
    def backbone_feature_dim(self) -> int:
        return int(self.frame_backbone.feature_dim)

    @property
    def backbone_mode(self) -> str:
        return "frame-2d"

    def forward(self, clips: torch.Tensor) -> VideoAnomalyOutput:
        frame_features = self.frame_backbone(clips)
        clip_embedding = self.temporal_encoder(frame_features)
        svdd_embedding = self.svdd_head(clip_embedding)
        return VideoAnomalyOutput(
            frame_features=frame_features,
            clip_embedding=clip_embedding,
            svdd_embedding=svdd_embedding,
            anomaly_score=self.svdd_head.score(svdd_embedding),
        )


class VideoAnomaly3DModel(nn.Module):
    """Clip-native 3D image-family backbone followed directly by Deep SVDD."""

    def __init__(
        self,
        backbone: InflatedImageBackbone3D,
        svdd_head: DeepSVDDHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.svdd_head = svdd_head

    @property
    def backbone_feature_dim(self) -> int:
        return int(self.backbone.feature_dim)

    @property
    def backbone_mode(self) -> str:
        return "3d"

    def forward(self, clips: torch.Tensor) -> VideoAnomalyOutput:
        clip_embedding = self.backbone(clips)
        svdd_embedding = self.svdd_head(clip_embedding)
        return VideoAnomalyOutput(
            frame_features=None,
            clip_embedding=clip_embedding,
            svdd_embedding=svdd_embedding,
            anomaly_score=self.svdd_head.score(svdd_embedding),
        )


def build_video_anomaly_model(
    model_name: str,
    config: dict[str, Any],
    *,
    backbone_factory: ImageFeatureBackboneFactory = build_default_frame_backbone,
    backbone_3d_factory=build_spatiotemporal_backbone_3d,
) -> VideoAnomalyModel | VideoAnomaly3DModel:
    backbone_mode = str(config.get("backbone_mode", "3d"))
    if backbone_mode == "3d":
        clip_length = int(config.get("clip_length", 16))
        temporal_kernel = int(config.get("backbone_temporal_kernel_size", 3))
        if clip_length < temporal_kernel:
            raise MLXUserError(
                "--clip-length must be at least --backbone-temporal-kernel-size "
                "for a 3D backbone."
            )
        backbone = backbone_3d_factory(model_name, config)
        return VideoAnomaly3DModel(
            backbone,
            DeepSVDDHead(
                backbone.feature_dim,
                int(config.get("svdd_hidden_dim", 256)),
                int(config.get("svdd_dim", 128)),
            ),
        )
    if backbone_mode != "frame-2d":
        raise MLXUserError("--backbone-mode must be '3d' or 'frame-2d'.")
    frame_backbone = FrameBackbone(backbone_factory(model_name, config))
    temporal = build_temporal_encoder(
        str(config.get("temporal_model", "tcn")),
        input_dim=frame_backbone.feature_dim,
        hidden_dim=int(config.get("temporal_hidden_dim", 256)),
        embedding_dim=int(config.get("temporal_embedding_dim", 128)),
        kernel_size=int(config.get("temporal_kernel_size", 3)),
        dropout=float(config.get("temporal_dropout", 0.0)),
    )
    return VideoAnomalyModel(
        frame_backbone,
        temporal,
        DeepSVDDHead(
            temporal.output_dim,
            int(config.get("svdd_hidden_dim", 256)),
            int(config.get("svdd_dim", 128)),
        ),
    )


__all__ = [
    "VideoAnomalyModel",
    "VideoAnomaly3DModel",
    "VideoAnomalyOutput",
    "build_video_anomaly_model",
]
