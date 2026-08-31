from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import torch

from mlx.core.deep_svdd import quantile_threshold
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.backbones import build_image_backbone
from mlx.modes.image_recognition_oc.models import DeepSVDDImageRecognizer


class OneClassAlgorithm(Protocol):
    name: str

    def build_model(self, backbone_name: str, config: dict[str, Any]): ...
    def initialize(self, model, loader, device: str) -> None: ...
    def training_step(self, model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def scores(self, model, images: torch.Tensor) -> torch.Tensor: ...
    def calibrate(self, model, scores: torch.Tensor, config: dict[str, Any]) -> float: ...
    def checkpoint_metadata(self, model, config: dict[str, Any]) -> dict[str, Any]: ...
    def config_from_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]: ...
    def resume_signature(self, config: dict[str, Any]) -> dict[str, Any]: ...
    def validate_loaded_model(self, model, checkpoint: dict[str, Any]) -> None: ...


class DeepSVDDAlgorithm:
    name = "deep-svdd"

    def __init__(self, backbone_factory=build_image_backbone) -> None:
        self.backbone_factory = backbone_factory

    def build_model(self, backbone_name: str, config: dict[str, Any]):
        self._validate(config)
        backbone = self.backbone_factory(backbone_name, config)
        return DeepSVDDImageRecognizer(
            backbone,
            hidden_dim=int(config.get("svdd_hidden_dim", 256)),
            embedding_dim=int(config.get("svdd_dim", 128)),
        )

    @torch.no_grad()
    def initialize(self, model, loader, device: str) -> None:
        was_training = model.training
        model.eval()
        total = torch.zeros_like(model.center, device=device)
        count = 0
        try:
            for images, _, _ in loader:
                embeddings = model(images.to(device)).embedding
                total += embeddings.sum(dim=0)
                count += embeddings.shape[0]
        finally:
            if was_training:
                model.train()
        if count == 0:
            raise MLXUserError("Cannot initialize Deep SVDD from an empty normal training set.")
        center = total / count
        near_zero = center.abs() < 0.1
        signs = torch.where(center < 0, -torch.ones_like(center), torch.ones_like(center))
        model.center.copy_(torch.where(near_zero, signs * 0.1, center))

    def training_step(self, model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = model(images).anomaly_score
        return scores.mean(), scores

    def scores(self, model, images: torch.Tensor) -> torch.Tensor:
        return model(images).anomaly_score

    def calibrate(self, model, scores: torch.Tensor, config: dict[str, Any]) -> float:
        threshold = quantile_threshold(scores, float(config.get("svdd_quantile", 0.95)))
        model.threshold.copy_(threshold.to(model.threshold.device))
        return float(model.threshold.item())

    def checkpoint_metadata(self, model, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "svdd_dim": int(config.get("svdd_dim", 128)),
            "svdd_hidden_dim": int(config.get("svdd_hidden_dim", 256)),
            "svdd_quantile": float(config.get("svdd_quantile", 0.95)),
            "center": model.center.detach().cpu(),
            "threshold": (
                float(model.threshold.item()) if torch.isfinite(model.threshold) else None
            ),
            "score_type": "squared_euclidean",
        }

    def config_from_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        return {
            "svdd_dim": int(checkpoint["svdd_dim"]),
            "svdd_hidden_dim": int(checkpoint["svdd_hidden_dim"]),
            "svdd_quantile": float(checkpoint["svdd_quantile"]),
        }

    def resume_signature(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "svdd_dim": int(config.get("svdd_dim", 128)),
            "svdd_hidden_dim": int(config.get("svdd_hidden_dim", 256)),
        }

    def validate_loaded_model(self, model, checkpoint: dict[str, Any]) -> None:
        if not torch.all(torch.isfinite(model.center)):
            raise ValueError("the stored Deep SVDD center contains non-finite values")
        metadata_center = torch.as_tensor(checkpoint["center"])
        if not torch.equal(metadata_center.cpu(), model.center.detach().cpu()):
            raise ValueError("checkpoint center metadata does not match model state")
        stored_threshold = checkpoint.get("threshold")
        state_threshold = float(model.threshold.item())
        if stored_threshold is None:
            if torch.isfinite(model.threshold):
                raise ValueError("checkpoint threshold metadata does not match model state")
        elif not torch.isfinite(model.threshold) or state_threshold != float(stored_threshold):
            raise ValueError("checkpoint threshold metadata does not match model state")

    @staticmethod
    def _validate(config: dict[str, Any]) -> None:
        if int(config.get("svdd_dim", 128)) < 1:
            raise MLXUserError("--svdd-dim must be greater than zero.")
        if int(config.get("svdd_hidden_dim", 256)) < 1:
            raise MLXUserError("--svdd-hidden-dim must be greater than zero.")
        quantile = float(config.get("svdd_quantile", 0.95))
        if not 0 < quantile < 1:
            raise MLXUserError("--svdd-quantile must be strictly between zero and one.")


@dataclass(frozen=True)
class OneClassAlgorithmRegistry:
    algorithms: Mapping[str, OneClassAlgorithm] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithms", MappingProxyType(dict(self.algorithms)))

    def register(self, algorithm: OneClassAlgorithm) -> "OneClassAlgorithmRegistry":
        name = algorithm.name.strip().lower()
        if not name:
            raise ValueError("One-class algorithm name cannot be empty.")
        algorithms = dict(self.algorithms)
        algorithms[name] = algorithm
        return OneClassAlgorithmRegistry(algorithms)

    def get(self, name: str) -> OneClassAlgorithm:
        algorithm = self.algorithms.get(name)
        if algorithm is None:
            available = ", ".join(sorted(self.algorithms))
            raise MLXUserError(
                f"Unsupported one-class model '{name}'. Available models: {available}."
            )
        return algorithm


DEFAULT_ALGORITHM_REGISTRY = OneClassAlgorithmRegistry().register(DeepSVDDAlgorithm())


__all__ = [
    "DEFAULT_ALGORITHM_REGISTRY",
    "DeepSVDDAlgorithm",
    "OneClassAlgorithm",
    "OneClassAlgorithmRegistry",
]
