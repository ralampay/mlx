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


def compute_binary_metrics(
    labels,
    scores,
    *,
    threshold: float,
) -> tuple[dict[str, float | int], dict[str, list[dict[str, float]]], np.ndarray]:
    targets = np.asarray(labels, dtype=np.int64)
    anomaly_scores = np.asarray(scores, dtype=np.float64)
    if targets.size == 0:
        raise MLXUserError("Cannot benchmark an empty prediction set.")
    if not np.all(np.isin(targets, (0, 1))):
        raise MLXUserError("Video anomaly labels must use 0=normal and 1=anomaly.")
    if not np.all(np.isfinite(anomaly_scores)) or not np.isfinite(threshold):
        raise MLXUserError("Video anomaly scores and the stored threshold must be finite.")
    if np.unique(targets).size < 2:
        raise MLXUserError(
            "Video anomaly benchmark requires both normal and anomaly samples for AUROC/AUPRC."
        )
    predictions = (anomaly_scores > threshold).astype(np.int64)
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1 = _divide(2 * precision * recall, precision + recall)
    fpr = _divide(fp, fp + tn)
    fnr = _divide(fn, fn + tp)
    normal_scores = anomaly_scores[targets == 0]
    anomaly_only_scores = anomaly_scores[targets == 1]
    metrics: dict[str, float | int] = {
        "normal_samples": int(normal_scores.size),
        "anomaly_samples": int(anomaly_only_scores.size),
        "auroc": float(roc_auc_score(targets, anomaly_scores)),
        "average_precision": float(average_precision_score(targets, anomaly_scores)),
        "auprc": float(average_precision_score(targets, anomaly_scores)),
        "normal_acceptance_rate": specificity,
        "specificity": specificity,
        "anomaly_detection_rate": recall,
        "sensitivity": recall,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "threshold": float(threshold),
        "normal_score_mean": float(normal_scores.mean()),
        "normal_score_std": float(normal_scores.std()),
        "anomaly_score_mean": float(anomaly_only_scores.mean()),
        "anomaly_score_std": float(anomaly_only_scores.std()),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(targets, anomaly_scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(targets, anomaly_scores)
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


__all__ = ["compute_binary_metrics", "prefixed_metrics"]
