from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from mlx.core.exceptions import MLXUserError


def compute_binary_score_metrics(
    labels,
    scores,
    *,
    threshold: float,
    task_label: str = "Binary benchmark",
    label_semantics_message: str | None = None,
    finite_message: str | None = None,
    both_classes_message: str | None = None,
) -> tuple[dict[str, float | int], dict[str, list[dict[str, float]]], np.ndarray]:
    """Evaluate higher-is-positive scores at a fixed deployment threshold."""

    targets = np.asarray(labels, dtype=np.int64)
    positive_scores = np.asarray(scores, dtype=np.float64)
    if targets.size == 0:
        raise MLXUserError("Cannot benchmark an empty prediction set.")
    if not np.all(np.isin(targets, (0, 1))):
        raise MLXUserError(
            label_semantics_message
            or f"{task_label} labels must use 0=normal and 1=anomaly."
        )
    if not np.all(np.isfinite(positive_scores)) or not np.isfinite(threshold):
        raise MLXUserError(
            finite_message
            or f"{task_label} scores and the stored threshold must be finite."
        )
    if np.unique(targets).size < 2:
        raise MLXUserError(
            both_classes_message
            or f"{task_label} requires both normal and anomaly samples for AUROC/AUPRC."
        )

    predictions = (positive_scores > threshold).astype(np.int64)
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1 = _divide(2 * precision * recall, precision + recall)
    normal_scores = positive_scores[targets == 0]
    anomaly_scores = positive_scores[targets == 1]
    average_precision = float(average_precision_score(targets, positive_scores))
    metrics: dict[str, float | int] = {
        "normal_samples": int(normal_scores.size),
        "anomaly_samples": int(anomaly_scores.size),
        "auroc": float(roc_auc_score(targets, positive_scores)),
        "average_precision": average_precision,
        "auprc": average_precision,
        "normal_acceptance_rate": specificity,
        "specificity": specificity,
        "anomaly_detection_rate": recall,
        "sensitivity": recall,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "false_positive_rate": _divide(fp, fp + tn),
        "false_negative_rate": _divide(fn, fn + tp),
        "threshold": float(threshold),
        "normal_score_mean": float(normal_scores.mean()),
        "normal_score_std": float(normal_scores.std()),
        "anomaly_score_mean": float(anomaly_scores.mean()),
        "anomaly_score_std": float(anomaly_scores.std()),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(targets, positive_scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(targets, positive_scores)
    curves = {
        "roc": [
            {
                "false_positive_rate": float(x),
                "true_positive_rate": float(y),
                "threshold": float(t),
            }
            for x, y, t in zip(roc_fpr, roc_tpr, roc_thresholds, strict=True)
        ],
        "precision_recall": [
            {
                "recall": float(r),
                "precision": float(p),
                "threshold": float(t) if np.isfinite(t) else float("nan"),
            }
            for r, p, t in zip(
                pr_recall,
                pr_precision,
                np.append(pr_thresholds, np.nan),
                strict=True,
            )
        ],
    }
    return metrics, curves, np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = ["compute_binary_score_metrics", "prefixed_metrics"]
