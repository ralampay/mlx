from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import nn
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.definitions import load_definition


@runtime_checkable
class ReconstructionLossDefinition(Protocol):
    name: str
    description: str

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        ...


class MSELossDefinition:
    name = "mse"
    description = "Mean squared reconstruction error."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        _reject_options(self.name, config)
        return nn.MSELoss()


class MAELossDefinition:
    name = "mae"
    description = "Mean absolute reconstruction error."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        _reject_options(self.name, config)
        return nn.L1Loss()


class SmoothL1LossDefinition:
    name = "smooth-l1"
    description = "Robust Smooth L1 reconstruction error; accepts positive beta."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        unknown = set(config) - {"beta"}
        if unknown:
            raise MLXUserError(f"Unsupported smooth-l1 option(s): {', '.join(sorted(unknown))}.")
        beta = float(config.get("beta", 1.0))
        if beta <= 0:
            raise MLXUserError("smooth-l1 beta must be greater than zero.")
        return nn.SmoothL1Loss(beta=beta)


class SimilarityPreservingLoss(nn.Module):
    """Match off-diagonal cosine similarities of corresponding embedding batches."""

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or z.ndim != 2 or x.shape[0] != z.shape[0]:
            raise ValueError("Similarity loss requires 2D batches with matching rows.")
        if x.shape[0] < 2 or min(x.shape[1], z.shape[1]) < 1:
            raise ValueError("Similarity loss requires at least two non-empty vectors.")
        if x.device != z.device or not x.is_floating_point() or not z.is_floating_point():
            raise ValueError("Similarity loss requires floating tensors on the same device.")
        # Disable autocast for the Gram products as well as the normalization.
        with torch.autocast(device_type=z.device.type, enabled=False):
            target = x.detach()
            target = target.float() if target.dtype in (torch.float16, torch.bfloat16) else target
            latent = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
            target = F.normalize(target, dim=1, eps=1e-8)
            latent = F.normalize(latent, dim=1, eps=1e-8)
            target_similarity = target @ target.T
            latent_similarity = latent @ latent.T
            mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
            return (latent_similarity[mask] - target_similarity[mask]).square().mean()


class MSESimilarityLoss(nn.Module):
    """Reconstruction MSE plus cosine geometry preservation at the bottleneck."""

    def __init__(self, similarity_weight: float = 1.0) -> None:
        super().__init__()
        message = "similarity_weight must be a finite, nonnegative number."
        if isinstance(similarity_weight, bool) or not isinstance(similarity_weight, (int, float)):
            raise MLXUserError(message)
        try:
            weight = float(similarity_weight)
        except OverflowError as exc:
            raise MLXUserError(message) from exc
        if not math.isfinite(weight) or weight < 0:
            raise MLXUserError(message)
        self.similarity_weight = weight
        self.requires_latent = self.similarity_weight > 0
        self.minimum_batch_size = 2 if self.requires_latent else 1
        self.similarity = SimilarityPreservingLoss()

    def forward(self, reconstruction, target, *, latent=None):
        reconstruction_loss = F.mse_loss(reconstruction, target)
        if not self.requires_latent:
            return reconstruction_loss
        if latent is None:
            raise ValueError("mse-similarity requires latent embeddings when its weight is positive.")
        return reconstruction_loss + self.similarity_weight * self.similarity(target, latent)


class MSESimilarityLossDefinition:
    name = "mse-similarity"
    description = "Reconstruction MSE plus latent cosine-similarity preservation."
    default_config = MappingProxyType({"similarity_weight": 1.0})

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        unknown = set(config) - set(self.default_config)
        if unknown:
            raise MLXUserError(f"Unsupported mse-similarity option(s): {', '.join(sorted(unknown))}.")
        return MSESimilarityLoss(config.get("similarity_weight", 1.0))


def _reject_options(name: str, config: Mapping[str, Any]) -> None:
    if config:
        raise MLXUserError(f"Loss '{name}' does not accept configuration options.")


BUILTIN_LOSSES: Mapping[str, str] = MappingProxyType(
    {
        "mae": "mlx.modes.autoencoder.losses:MAELossDefinition",
        "mse": "mlx.modes.autoencoder.losses:MSELossDefinition",
        "mse-similarity": "mlx.modes.autoencoder.losses:MSESimilarityLossDefinition",
        "smooth-l1": "mlx.modes.autoencoder.losses:SmoothL1LossDefinition",
    }
)


@dataclass(frozen=True)
class ReconstructionLossRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: BUILTIN_LOSSES)
    descriptions: Mapping[str, str] = field(default_factory=lambda: {
        "mae": MAELossDefinition.description, "mse": MSELossDefinition.description,
        "smooth-l1": SmoothL1LossDefinition.description,
        "mse-similarity": MSESimilarityLossDefinition.description,
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptions", MappingProxyType(dict(self.descriptions)))
        object.__setattr__(
            self,
            "entries",
            MappingProxyType({str(key).strip().lower(): str(value) for key, value in self.entries.items()}),
        )

    def register(self, name: str, definition_path: str, *, description: str = "") -> "ReconstructionLossRegistry":
        normalized = name.strip().lower()
        if not normalized or ":" not in definition_path:
            raise ValueError("Loss registration requires a name and package.module:DefinitionClass path.")
        return ReconstructionLossRegistry({**self.entries, normalized: definition_path}, {**self.descriptions, normalized: description})

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def resolve(self, reference: str) -> tuple[ReconstructionLossDefinition, str]:
        requested = reference.strip()
        path = requested if ":" in requested else self.entries.get(requested.lower())
        if path is None:
            available = ", ".join(self.names()) or "none"
            raise MLXUserError(
                f"Unsupported autoencoder loss '{reference}'. Available losses: {available}; "
                "external losses may use package.module:DefinitionClass."
            )
        definition = load_definition(path, "loss")
        if not isinstance(definition, ReconstructionLossDefinition):
            raise MLXUserError(
                f"Loss definition '{path}' must provide name, description, and build(config)."
            )
        return definition, path


DEFAULT_LOSS_REGISTRY = ReconstructionLossRegistry()


__all__ = [
    "DEFAULT_LOSS_REGISTRY",
    "MAELossDefinition",
    "MSELossDefinition",
    "MSESimilarityLoss",
    "MSESimilarityLossDefinition",
    "SimilarityPreservingLoss",
    "ReconstructionLossDefinition",
    "ReconstructionLossRegistry",
    "SmoothL1LossDefinition",
]
