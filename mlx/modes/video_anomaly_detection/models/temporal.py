from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError


class TemporalEncoder(Protocol):
    output_dim: int

    def __call__(self, frame_features: torch.Tensor) -> torch.Tensor: ...


class TemporalConvEncoder(nn.Module):
    """Simple replaceable temporal CNN over frame feature vectors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, embedding_dim) < 1:
            raise MLXUserError("Temporal input, hidden, and embedding dimensions must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise MLXUserError("--temporal-kernel-size must be a positive odd integer.")
        if not 0 <= dropout < 1:
            raise MLXUserError("--temporal-dropout must be in [0, 1).")
        padding = kernel_size // 2
        self.output_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError("Temporal encoder expects [B, T, D] frame features.")
        return self.network(frame_features.transpose(1, 2))


TemporalBuilder = Callable[..., nn.Module]


@dataclass(frozen=True)
class TemporalEncoderRegistry:
    entries: Mapping[str, TemporalBuilder] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def register(self, name: str, builder: TemporalBuilder) -> "TemporalEncoderRegistry":
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Temporal encoder name cannot be empty.")
        entries = dict(self.entries)
        entries[normalized] = builder
        return TemporalEncoderRegistry(entries)


DEFAULT_TEMPORAL_ENCODER_REGISTRY = TemporalEncoderRegistry({"tcn": TemporalConvEncoder})
TEMPORAL_ENCODERS = DEFAULT_TEMPORAL_ENCODER_REGISTRY.entries


def build_temporal_encoder(
    name: str,
    *,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    kernel_size: int,
    dropout: float,
    registry: TemporalEncoderRegistry | None = None,
) -> nn.Module:
    entries = (registry or DEFAULT_TEMPORAL_ENCODER_REGISTRY).entries
    builder = entries.get(name)
    if builder is None:
        available = ", ".join(sorted(entries))
        raise MLXUserError(
            f"Unsupported temporal model '{name}'. Available models: {available}."
        )
    return builder(
        input_dim,
        hidden_dim,
        embedding_dim,
        kernel_size=kernel_size,
        dropout=dropout,
    )


__all__ = [
    "TEMPORAL_ENCODERS",
    "DEFAULT_TEMPORAL_ENCODER_REGISTRY",
    "TemporalEncoderRegistry",
    "TemporalConvEncoder",
    "TemporalEncoder",
    "build_temporal_encoder",
]
