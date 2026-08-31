from __future__ import annotations

from typing import Any

import numpy as np

from mlx.core.binary_metrics import compute_binary_score_metrics


def compute_binary_metrics(
    labels,
    scores,
    *,
    threshold: float,
) -> tuple[dict[str, float | int], dict[str, list[dict[str, float]]], np.ndarray]:
    return compute_binary_score_metrics(
        labels,
        scores,
        threshold=threshold,
        task_label="Video anomaly benchmark",
        label_semantics_message="Video anomaly labels must use 0=normal and 1=anomaly.",
        finite_message="Video anomaly scores and the stored threshold must be finite.",
        both_classes_message=(
            "Video anomaly benchmark requires both normal and anomaly samples for AUROC/AUPRC."
        ),
    )


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


__all__ = ["compute_binary_metrics", "prefixed_metrics"]
