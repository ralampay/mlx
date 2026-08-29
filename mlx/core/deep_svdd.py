from __future__ import annotations

import torch

from mlx.core.exceptions import MLXUserError


def squared_euclidean_score(
    embedding: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    """Return the Deep-SVDD squared distance for each batch item."""

    return torch.sum((embedding - center) ** 2, dim=1)


def quantile_threshold(scores: torch.Tensor, quantile: float) -> torch.Tensor:
    if not 0 < quantile < 1:
        raise MLXUserError(
            "The Deep SVDD calibration quantile must be strictly between zero and one."
        )
    if scores.numel() == 0:
        raise MLXUserError(
            "Cannot calibrate Deep SVDD from an empty score collection."
        )
    return torch.quantile(scores.float(), quantile)


__all__ = ["quantile_threshold", "squared_euclidean_score"]
