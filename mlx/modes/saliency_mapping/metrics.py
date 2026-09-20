from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BETA_SQUARED = 0.3


def mean_absolute_error(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probabilities, targets = _validated_arrays(probabilities, targets)
    return float(np.abs(probabilities - targets).mean())


def f_beta(precision: float, recall: float, *, beta_squared: float = BETA_SQUARED) -> float:
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return float((1.0 + beta_squared) * precision * recall / denominator)


def saliency_threshold_sweep(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold_steps: int = 101,
    target_threshold: float = 0.5,
    beta_squared: float = BETA_SQUARED,
) -> list[dict[str, float]]:
    if threshold_steps < 2:
        raise ValueError("threshold_steps must be at least 2.")
    probabilities, targets = _validated_arrays(probabilities, targets)
    truth = targets.reshape(-1) >= target_threshold
    scores = probabilities.reshape(-1)
    rows = []
    for threshold in np.linspace(0.0, 1.0, threshold_steps):
        predicted = scores >= threshold
        tp = float(np.count_nonzero(predicted & truth))
        fp = float(np.count_nonzero(predicted & ~truth))
        fn = float(np.count_nonzero(~predicted & truth))
        precision = tp / (tp + fp) if tp + fp else (1.0 if not truth.any() else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f_beta": f_beta(precision, recall, beta_squared=beta_squared),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
            }
        )
    return rows


def summarize_threshold_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("At least one threshold row is required.")
    best = max(rows, key=lambda row: (row["f_beta"], -row["threshold"]))
    return {
        "max_f_beta": float(best["f_beta"]),
        "mean_f_beta": float(np.mean([row["f_beta"] for row in rows])),
        "best_threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
    }


def per_image_saliency_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold_steps: int = 101,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = saliency_threshold_sweep(
        probabilities,
        targets,
        threshold_steps=threshold_steps,
    )
    return {"mae": mean_absolute_error(probabilities, targets), **summarize_threshold_rows(rows)}, rows


@dataclass
class SaliencyMetricAccumulator:
    threshold_steps: int = 101

    def __post_init__(self) -> None:
        if self.threshold_steps < 2:
            raise ValueError("threshold_steps must be at least 2.")
        self.thresholds = np.linspace(0.0, 1.0, self.threshold_steps)
        self.true_positive = np.zeros(self.threshold_steps, dtype=np.int64)
        self.false_positive = np.zeros(self.threshold_steps, dtype=np.int64)
        self.false_negative = np.zeros(self.threshold_steps, dtype=np.int64)
        self.absolute_error_sum = 0.0
        self.pixel_count = 0
        self.image_count = 0

    def update(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        probabilities, targets = _validated_arrays(probabilities, targets)
        scores = probabilities.reshape(-1)
        truth = targets.reshape(-1) >= 0.5
        self.absolute_error_sum += float(np.abs(scores - targets.reshape(-1)).sum())
        self.pixel_count += scores.size
        self.image_count += int(probabilities.shape[0]) if probabilities.ndim >= 3 else 1
        for index, threshold in enumerate(self.thresholds):
            predicted = scores >= threshold
            self.true_positive[index] += np.count_nonzero(predicted & truth)
            self.false_positive[index] += np.count_nonzero(predicted & ~truth)
            self.false_negative[index] += np.count_nonzero(~predicted & truth)

    def finalize(self) -> tuple[dict[str, float], list[dict[str, float]]]:
        rows = []
        truth_count = int((self.true_positive + self.false_negative).max(initial=0))
        for index, threshold in enumerate(self.thresholds):
            tp = float(self.true_positive[index])
            fp = float(self.false_positive[index])
            fn = float(self.false_negative[index])
            precision = tp / (tp + fp) if tp + fp else (1.0 if truth_count == 0 else 0.0)
            recall = tp / (tp + fn) if tp + fn else 1.0
            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f_beta": f_beta(precision, recall),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                }
            )
        metrics = summarize_threshold_rows(rows)
        metrics.update(
            {
                "mae": self.absolute_error_sum / max(1, self.pixel_count),
                "evaluated_images": float(self.image_count),
                "evaluated_pixels": float(self.pixel_count),
                "beta_squared": BETA_SQUARED,
            }
        )
        return metrics, rows


def _validated_arrays(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError("Saliency probabilities and targets must have matching shapes.")
    if probabilities.size == 0:
        raise ValueError("Saliency metrics require at least one value.")
    if not np.isfinite(probabilities).all() or not np.isfinite(targets).all():
        raise ValueError("Saliency probabilities and targets must be finite.")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("Saliency probabilities must be in [0, 1].")
    if targets.min() < 0 or targets.max() > 1:
        raise ValueError("Saliency targets must be in [0, 1].")
    return probabilities, targets


__all__ = [
    "BETA_SQUARED",
    "SaliencyMetricAccumulator",
    "f_beta",
    "mean_absolute_error",
    "per_image_saliency_metrics",
    "saliency_threshold_sweep",
    "summarize_threshold_rows",
]
