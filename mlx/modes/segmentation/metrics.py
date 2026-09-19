from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def metric_slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def nanmean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def weighted_nanmean(values: list[float], weights: list[float]) -> float:
    valid = [
        (value, weight)
        for value, weight in zip(values, weights, strict=True)
        if np.isfinite(value) and weight > 0
    ]
    if not valid:
        return float("nan")
    total = sum(weight for _, weight in valid)
    return float(sum(value * weight for value, weight in valid) / total)


def confusion_matrix_from_arrays(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    targets = targets.reshape(-1).astype(np.int64, copy=False)
    predictions = predictions.reshape(-1).astype(np.int64, copy=False)
    valid = (
        (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )
    encoded = targets[valid] * num_classes + predictions[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def class_metrics_from_confusion(
    matrix: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    total = float(matrix.sum())
    rows: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(class_names):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        tn = total - tp - fp - fn
        support = tp + fn
        predicted_support = tp + fp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        npv = safe_divide(tn, tn + fn)
        dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        iou = safe_divide(tp, tp + fp + fn)
        mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = (
            (tp * tn - fp * fn) / np.sqrt(mcc_denominator)
            if mcc_denominator > 0
            else float("nan")
        )
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn),
                "support": int(support),
                "predicted_support": int(predicted_support),
                "prevalence": safe_divide(support, total),
                "predicted_prevalence": safe_divide(predicted_support, total),
                "precision": precision,
                "ppv": precision,
                "recall": recall,
                "sensitivity": recall,
                "specificity": specificity,
                "npv": npv,
                "false_positive_rate": safe_divide(fp, fp + tn),
                "false_negative_rate": safe_divide(fn, fn + tp),
                "false_discovery_rate": safe_divide(fp, fp + tp),
                "false_omission_rate": safe_divide(fn, fn + tn),
                "balanced_accuracy": nanmean([recall, specificity]),
                "f1": dice,
                "dice": dice,
                "iou": iou,
                "jaccard": iou,
                "one_vs_rest_accuracy": safe_divide(tp + tn, total),
                "mcc": float(mcc),
                "area_difference": float(predicted_support - support),
                "absolute_area_error": float(abs(predicted_support - support)),
                "relative_area_error": safe_divide(predicted_support - support, support),
                "absolute_relative_area_error": safe_divide(
                    abs(predicted_support - support),
                    support,
                ),
            }
        )
    return rows


def aggregate_confusion_metrics(
    matrix: np.ndarray,
    class_rows: list[dict[str, Any]],
) -> dict[str, float]:
    total = float(matrix.sum())
    supports = [float(row["support"]) for row in class_rows]
    precision = [float(row["precision"]) for row in class_rows]
    recall = [float(row["recall"]) for row in class_rows]
    specificity = [float(row["specificity"]) for row in class_rows]
    dice = [float(row["dice"]) for row in class_rows]
    iou = [float(row["iou"]) for row in class_rows]

    tp_sum = float(np.trace(matrix))
    fp_sum = float(sum(float(row["false_positive"]) for row in class_rows))
    fn_sum = float(sum(float(row["false_negative"]) for row in class_rows))
    micro_precision = safe_divide(tp_sum, tp_sum + fp_sum)
    micro_recall = safe_divide(tp_sum, tp_sum + fn_sum)
    micro_dice = safe_divide(2.0 * tp_sum, 2.0 * tp_sum + fp_sum + fn_sum)
    micro_iou = safe_divide(tp_sum, tp_sum + fp_sum + fn_sum)

    foreground_rows = class_rows[1:] if len(class_rows) > 1 else class_rows
    foreground_dice = [float(row["dice"]) for row in foreground_rows]
    foreground_iou = [float(row["iou"]) for row in foreground_rows]
    observed = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    expected_accuracy = safe_divide(float(np.dot(observed, predicted)), total * total)
    pixel_accuracy = safe_divide(tp_sum, total)
    kappa = safe_divide(pixel_accuracy - expected_accuracy, 1.0 - expected_accuracy)

    generalized_denominator = 0.0
    generalized_numerator = 0.0
    for index, support in enumerate(supports):
        if support <= 0:
            continue
        weight = 1.0 / (support * support)
        tp = float(matrix[index, index])
        predicted_support = float(matrix[:, index].sum())
        generalized_numerator += 2.0 * weight * tp
        generalized_denominator += weight * (support + predicted_support)

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_class_accuracy": nanmean(recall),
        "macro_precision": nanmean(precision),
        "micro_precision": micro_precision,
        "weighted_precision": weighted_nanmean(precision, supports),
        "macro_recall": nanmean(recall),
        "micro_recall": micro_recall,
        "weighted_recall": weighted_nanmean(recall, supports),
        "macro_specificity": nanmean(specificity),
        "weighted_specificity": weighted_nanmean(specificity, supports),
        "macro_f1": nanmean(dice),
        "micro_f1": micro_dice,
        "weighted_f1": weighted_nanmean(dice, supports),
        "macro_dice": nanmean(dice),
        "micro_dice": micro_dice,
        "weighted_dice": weighted_nanmean(dice, supports),
        "mean_iou": nanmean(iou),
        "micro_iou": micro_iou,
        "weighted_iou": weighted_nanmean(iou, supports),
        "frequency_weighted_iou": weighted_nanmean(iou, supports),
        "mean_foreground_dice": nanmean(foreground_dice),
        "mean_foreground_iou": nanmean(foreground_iou),
        "generalized_dice": safe_divide(generalized_numerator, generalized_denominator),
        "cohen_kappa": kappa,
        "valid_class_count": float(sum(np.isfinite(value) for value in dice)),
        "valid_foreground_class_count": float(
            sum(np.isfinite(value) for value in foreground_dice)
        ),
    }


def multiclass_mcc_from_confusion(matrix: np.ndarray) -> float:
    """Compute multiclass MCC without retaining the original pixel labels."""
    sample_count = float(matrix.sum())
    correct = float(np.trace(matrix))
    true_totals = matrix.sum(axis=1).astype(np.float64, copy=False)
    predicted_totals = matrix.sum(axis=0).astype(np.float64, copy=False)
    covariance = correct * sample_count - float(
        np.dot(true_totals, predicted_totals)
    )
    true_covariance = sample_count**2 - float(np.dot(true_totals, true_totals))
    predicted_covariance = sample_count**2 - float(
        np.dot(predicted_totals, predicted_totals)
    )
    denominator = np.sqrt(true_covariance * predicted_covariance)
    return float(covariance / denominator) if denominator else 0.0


class StreamingSegmentationMetrics:
    """Accumulate dataset-wide segmentation metrics with bounded memory."""

    def __init__(
        self,
        *,
        num_classes: int,
        calibration_bins: int,
        curve_bins: int,
        threshold_steps: int,
        configured_threshold: float,
    ) -> None:
        self.num_classes = num_classes
        self.calibration_bins = calibration_bins
        self.curve_bins = curve_bins
        self.pixel_count = 0
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )
        self._negative_log_likelihood_sum = 0.0
        self._confidence_sum = 0.0
        self._entropy_sum = 0.0
        self._class_brier_sums = np.zeros(num_classes, dtype=np.float64)
        self._calibration_counts = np.zeros(calibration_bins, dtype=np.int64)
        self._calibration_confidence_sums = np.zeros(
            calibration_bins, dtype=np.float64
        )
        self._calibration_correct_counts = np.zeros(
            calibration_bins, dtype=np.int64
        )
        histogram_shape = (num_classes, curve_bins)
        self._positive_histograms = np.zeros(histogram_shape, dtype=np.int64)
        self._negative_histograms = np.zeros(histogram_shape, dtype=np.int64)
        self.thresholds = (
            np.asarray(
                sorted(
                    set(
                        np.linspace(0.0, 1.0, threshold_steps).tolist()
                        + [configured_threshold]
                    )
                ),
                dtype=np.float64,
            )
            if num_classes == 2
            else np.asarray([], dtype=np.float64)
        )
        self._threshold_true_positive = np.zeros(
            len(self.thresholds), dtype=np.int64
        )
        self._threshold_false_positive = np.zeros(
            len(self.thresholds), dtype=np.int64
        )
        self._binary_positive_count = 0
        self._binary_negative_count = 0

    def update(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        flattened_targets = targets.reshape(-1).astype(np.int64, copy=False)
        flattened_predictions = predictions.reshape(-1).astype(np.int64, copy=False)
        flattened_probabilities = probabilities.reshape(-1, self.num_classes)
        if len(flattened_targets) != len(flattened_probabilities):
            raise ValueError(
                "Segmentation targets and probabilities must have equal pixel counts."
            )

        batch_pixel_count = len(flattened_targets)
        self.pixel_count += batch_pixel_count
        self.confusion_matrix += confusion_matrix_from_arrays(
            flattened_targets,
            flattened_predictions,
            self.num_classes,
        )
        selected = np.clip(
            np.take_along_axis(
                flattened_probabilities,
                flattened_targets[:, None],
                axis=1,
            ).reshape(-1),
            1e-12,
            1.0,
        ).astype(np.float64, copy=False)
        self._negative_log_likelihood_sum += float(-np.log(selected).sum())
        confidence = flattened_probabilities.max(axis=1)
        probability_predictions = flattened_probabilities.argmax(axis=1)
        self._confidence_sum += float(confidence.sum(dtype=np.float64))

        calibration_indices = np.clip(
            np.searchsorted(
                np.linspace(0.0, 1.0, self.calibration_bins + 1),
                confidence,
                side="right",
            )
            - 1,
            0,
            self.calibration_bins - 1,
        )
        self._calibration_counts += np.bincount(
            calibration_indices, minlength=self.calibration_bins
        )
        self._calibration_confidence_sums += np.bincount(
            calibration_indices,
            weights=confidence,
            minlength=self.calibration_bins,
        )
        self._calibration_correct_counts += np.bincount(
            calibration_indices,
            weights=(probability_predictions == flattened_targets).astype(np.int64),
            minlength=self.calibration_bins,
        ).astype(np.int64)

        for class_index in range(self.num_classes):
            scores = flattened_probabilities[:, class_index]
            scores_float64 = scores.astype(np.float64, copy=False)
            binary_targets = flattened_targets == class_index
            self._class_brier_sums[class_index] += float(
                np.square(scores_float64 - binary_targets).sum()
            )
            self._entropy_sum += float(
                -np.sum(
                    scores_float64
                    * np.log(np.clip(scores_float64, 1e-12, 1.0))
                )
            )
            score_bins = np.minimum(
                (scores * self.curve_bins).astype(np.int64),
                self.curve_bins - 1,
            )
            self._positive_histograms[class_index] += np.bincount(
                score_bins[binary_targets], minlength=self.curve_bins
            )
            self._negative_histograms[class_index] += np.bincount(
                score_bins[~binary_targets], minlength=self.curve_bins
            )

        if self.num_classes == 2:
            self._update_threshold_counts(
                flattened_targets,
                flattened_probabilities[:, 1],
            )

    def finalize(
        self,
        class_names: list[str],
    ) -> tuple[
        dict[str, float],
        list[dict[str, Any]],
        dict[str, list[dict[str, float]]],
        list[dict[str, float]],
    ]:
        class_rows = class_metrics_from_confusion(self.confusion_matrix, class_names)
        metrics = aggregate_confusion_metrics(self.confusion_matrix, class_rows)
        metrics["multiclass_mcc"] = multiclass_mcc_from_confusion(
            self.confusion_matrix
        )
        metrics.update(self._probability_metrics(class_rows))
        curves = {
            "calibration": self._calibration_rows(),
            "roc": [],
            "precision_recall": [],
        }
        self._add_curve_metrics(metrics, class_rows, curves)
        threshold_rows: list[dict[str, float]] = []
        if self.num_classes == 2:
            threshold_rows, threshold_summary = _threshold_rows_from_counts(
                self.thresholds,
                self._threshold_true_positive,
                self._threshold_false_positive,
                positive_count=self._binary_positive_count,
                negative_count=self._binary_negative_count,
            )
            metrics.update(threshold_summary)
        return metrics, class_rows, curves, threshold_rows

    def _update_threshold_counts(
        self,
        targets: np.ndarray,
        foreground_scores: np.ndarray,
    ) -> None:
        binary_targets = targets > 0
        positive_scores = np.sort(foreground_scores[binary_targets])
        negative_scores = np.sort(foreground_scores[~binary_targets])
        self._binary_positive_count += len(positive_scores)
        self._binary_negative_count += len(negative_scores)
        self._threshold_true_positive += len(positive_scores) - np.searchsorted(
            positive_scores,
            self.thresholds,
            side="left",
        )
        self._threshold_false_positive += len(negative_scores) - np.searchsorted(
            negative_scores,
            self.thresholds,
            side="left",
        )

    def _probability_metrics(
        self,
        class_rows: list[dict[str, Any]],
    ) -> dict[str, float]:
        denominator = max(1, self.pixel_count)
        calibration_rows = self._calibration_rows()
        expected_error = sum(
            row["count"]
            / denominator
            * abs(row["accuracy"] - row["confidence"])
            for row in calibration_rows
        )
        metrics = {
            "negative_log_likelihood": self._negative_log_likelihood_sum / denominator,
            "multiclass_brier_score": float(self._class_brier_sums.sum())
            / denominator,
            "mean_confidence": self._confidence_sum / denominator,
            "mean_predictive_entropy": self._entropy_sum / denominator,
            "expected_calibration_error": float(expected_error),
            "maximum_calibration_error": float(
                max(
                    (
                        abs(row["accuracy"] - row["confidence"])
                        for row in calibration_rows
                    ),
                    default=0.0,
                )
            ),
            "curve_approximation_bins": float(self.curve_bins),
        }
        for class_index, class_row in enumerate(class_rows):
            slug = metric_slug(str(class_row["class_name"]))
            metrics[f"brier_{slug}"] = (
                float(self._class_brier_sums[class_index]) / denominator
            )
        return metrics

    def _calibration_rows(self) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for index, count in enumerate(self._calibration_counts):
            if count == 0:
                continue
            rows.append(
                {
                    "bin_lower": index / self.calibration_bins,
                    "bin_upper": (index + 1) / self.calibration_bins,
                    "count": float(count),
                    "confidence": float(
                        self._calibration_confidence_sums[index] / count
                    ),
                    "accuracy": float(
                        self._calibration_correct_counts[index] / count
                    ),
                }
            )
        return rows

    def _add_curve_metrics(
        self,
        metrics: dict[str, float],
        class_rows: list[dict[str, Any]],
        curves: dict[str, list[dict[str, float]]],
    ) -> None:
        auc_values: list[float] = []
        ap_values: list[float] = []
        weights: list[float] = []
        for class_index, class_row in enumerate(class_rows):
            positive = self._positive_histograms[class_index]
            negative = self._negative_histograms[class_index]
            positive_count = int(positive.sum())
            negative_count = int(negative.sum())
            slug = metric_slug(str(class_row["class_name"]))
            if positive_count == 0 or negative_count == 0:
                metrics[f"roc_auc_{slug}"] = float("nan")
                metrics[f"average_precision_{slug}"] = float("nan")
                metrics[f"pr_auc_{slug}"] = float("nan")
                continue

            occupied = np.flatnonzero((positive + negative) > 0)[::-1]
            true_positive = np.cumsum(positive[occupied], dtype=np.int64)
            false_positive = np.cumsum(negative[occupied], dtype=np.int64)
            true_positive_rate = true_positive / positive_count
            false_positive_rate = false_positive / negative_count
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive_rate
            roc_x = np.concatenate(([0.0], false_positive_rate))
            roc_y = np.concatenate(([0.0], true_positive_rate))
            pr_recall = np.concatenate(([0.0], recall))
            pr_precision = np.concatenate(([1.0], precision))
            roc_auc = float(np.trapezoid(roc_y, roc_x))
            average_precision = float(
                np.sum(np.diff(pr_recall) * pr_precision[1:])
            )
            pr_auc = float(np.trapezoid(pr_precision, pr_recall))
            metrics[f"roc_auc_{slug}"] = roc_auc
            metrics[f"average_precision_{slug}"] = average_precision
            metrics[f"pr_auc_{slug}"] = pr_auc
            class_row["roc_auc"] = roc_auc
            class_row["average_precision"] = average_precision
            class_row["pr_auc"] = pr_auc
            auc_values.append(roc_auc)
            ap_values.append(average_precision)
            weights.append(float(class_row["support"]))

            curves["roc"].append(
                {
                    "class_index": float(class_index),
                    "false_positive_rate": 0.0,
                    "true_positive_rate": 0.0,
                    "threshold": float("inf"),
                }
            )
            curves["precision_recall"].append(
                {
                    "class_index": float(class_index),
                    "recall": 0.0,
                    "precision": 1.0,
                    "threshold": float("inf"),
                }
            )
            for offset, bin_index in enumerate(occupied):
                threshold = float(bin_index / self.curve_bins)
                curves["roc"].append(
                    {
                        "class_index": float(class_index),
                        "false_positive_rate": float(false_positive_rate[offset]),
                        "true_positive_rate": float(true_positive_rate[offset]),
                        "threshold": threshold,
                    }
                )
                curves["precision_recall"].append(
                    {
                        "class_index": float(class_index),
                        "recall": float(recall[offset]),
                        "precision": float(precision[offset]),
                        "threshold": threshold,
                    }
                )

        metrics["macro_roc_auc"] = nanmean(auc_values)
        metrics["weighted_roc_auc"] = weighted_nanmean(auc_values, weights)
        metrics["macro_average_precision"] = nanmean(ap_values)
        metrics["weighted_average_precision"] = weighted_nanmean(
            ap_values, weights
        )


def probability_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_rows: list[dict[str, Any]],
    *,
    calibration_bins: int,
) -> tuple[dict[str, float], dict[str, list[dict[str, float]]]]:
    flattened_targets = targets.reshape(-1).astype(np.int64, copy=False)
    flattened_probabilities = probabilities.reshape(-1, probabilities.shape[-1]).astype(
        np.float64,
        copy=False,
    )
    selected = np.clip(
        flattened_probabilities[np.arange(len(flattened_targets)), flattened_targets],
        1e-12,
        1.0,
    )
    metrics: dict[str, float] = {
        "negative_log_likelihood": float(-np.log(selected).mean()),
        "multiclass_brier_score": float(
            np.mean(
                np.sum(flattened_probabilities**2, axis=1)
                - 2.0 * selected
                + 1.0
            )
        ),
        "mean_confidence": float(flattened_probabilities.max(axis=1).mean()),
        "mean_predictive_entropy": float(
            np.mean(
                -np.sum(
                    flattened_probabilities
                    * np.log(np.clip(flattened_probabilities, 1e-12, 1.0)),
                    axis=1,
                )
            )
        ),
    }
    calibration_rows = _calibration_rows(
        flattened_targets,
        flattened_probabilities,
        calibration_bins=calibration_bins,
    )
    total = max(1, len(flattened_targets))
    metrics["expected_calibration_error"] = float(
        sum(row["count"] / total * abs(row["accuracy"] - row["confidence"]) for row in calibration_rows)
    )
    metrics["maximum_calibration_error"] = float(
        max((abs(row["accuracy"] - row["confidence"]) for row in calibration_rows), default=0.0)
    )

    curves: dict[str, list[dict[str, float]]] = {"calibration": calibration_rows}
    roc_rows: list[dict[str, float]] = []
    pr_rows: list[dict[str, float]] = []
    auc_values: list[float] = []
    ap_values: list[float] = []
    weights: list[float] = []
    for class_index, class_row in enumerate(class_rows):
        binary_target = (flattened_targets == class_index).astype(np.uint8)
        scores = flattened_probabilities[:, class_index]
        slug = metric_slug(str(class_row["class_name"]))
        metrics[f"brier_{slug}"] = float(np.mean((scores - binary_target) ** 2))
        if np.unique(binary_target).size < 2:
            metrics[f"roc_auc_{slug}"] = float("nan")
            metrics[f"average_precision_{slug}"] = float("nan")
            metrics[f"pr_auc_{slug}"] = float("nan")
            continue
        roc_auc = float(roc_auc_score(binary_target, scores))
        average_precision = float(average_precision_score(binary_target, scores))
        fpr, tpr, roc_thresholds = roc_curve(binary_target, scores)
        precision, recall, pr_thresholds = precision_recall_curve(binary_target, scores)
        pr_auc = float(abs(np.trapezoid(precision, recall)))
        metrics[f"roc_auc_{slug}"] = roc_auc
        metrics[f"average_precision_{slug}"] = average_precision
        metrics[f"pr_auc_{slug}"] = pr_auc
        class_row["roc_auc"] = roc_auc
        class_row["average_precision"] = average_precision
        class_row["pr_auc"] = pr_auc
        auc_values.append(roc_auc)
        ap_values.append(average_precision)
        weights.append(float(class_row["support"]))
        roc_rows.extend(
            {
                "class_index": float(class_index),
                "false_positive_rate": float(x_value),
                "true_positive_rate": float(y_value),
                "threshold": float(threshold),
            }
            for x_value, y_value, threshold in zip(fpr, tpr, roc_thresholds, strict=True)
        )
        padded_thresholds = np.append(pr_thresholds, np.nan)
        pr_rows.extend(
            {
                "class_index": float(class_index),
                "recall": float(recall_value),
                "precision": float(precision_value),
                "threshold": float(threshold),
            }
            for recall_value, precision_value, threshold in zip(
                recall,
                precision,
                padded_thresholds,
                strict=True,
            )
        )
    metrics["macro_roc_auc"] = nanmean(auc_values)
    metrics["weighted_roc_auc"] = weighted_nanmean(auc_values, weights)
    metrics["macro_average_precision"] = nanmean(ap_values)
    metrics["weighted_average_precision"] = weighted_nanmean(ap_values, weights)
    curves["roc"] = roc_rows
    curves["precision_recall"] = pr_rows
    return metrics, curves


def _calibration_rows(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bins: int,
) -> list[dict[str, float]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(calibration_bins):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == calibration_bins - 1 else confidence < upper
        )
        if not selected.any():
            continue
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": float(selected.sum()),
                "confidence": float(confidence[selected].mean()),
                "accuracy": float(correct[selected].mean()),
            }
        )
    return rows


def boundary_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    tolerance: int,
) -> dict[str, float]:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    if not target.any() and not prediction.any():
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "surface_dice": 1.0,
            "hausdorff_distance": 0.0,
            "hausdorff_95": 0.0,
            "average_symmetric_surface_distance": 0.0,
        }
    if not target.any() or not prediction.any():
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "surface_dice": 0.0,
            "hausdorff_distance": float("nan"),
            "hausdorff_95": float("nan"),
            "average_symmetric_surface_distance": float("nan"),
        }

    target_boundary = _mask_boundary(target)
    prediction_boundary = _mask_boundary(prediction)
    kernel_size = tolerance * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    target_dilated = cv2.dilate(target_boundary.astype(np.uint8), kernel) > 0
    prediction_dilated = cv2.dilate(prediction_boundary.astype(np.uint8), kernel) > 0
    matched_prediction = float((prediction_boundary & target_dilated).sum())
    matched_target = float((target_boundary & prediction_dilated).sum())
    prediction_count = float(prediction_boundary.sum())
    target_count = float(target_boundary.sum())
    precision = safe_divide(matched_prediction, prediction_count)
    recall = safe_divide(matched_target, target_count)
    boundary_f1 = safe_divide(2.0 * precision * recall, precision + recall)
    surface_dice = safe_divide(
        matched_prediction + matched_target,
        prediction_count + target_count,
    )

    distance_to_target = cv2.distanceTransform(
        (~target_boundary).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )[prediction_boundary]
    distance_to_prediction = cv2.distanceTransform(
        (~prediction_boundary).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )[target_boundary]
    distances = np.concatenate((distance_to_target, distance_to_prediction))
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": boundary_f1,
        "surface_dice": surface_dice,
        "hausdorff_distance": float(distances.max()),
        "hausdorff_95": float(np.percentile(distances, 95)),
        "average_symmetric_surface_distance": float(
            (distance_to_target.mean() + distance_to_prediction.mean()) / 2.0
        ),
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    eroded = cv2.erode(
        mask_uint8,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (mask_uint8 - eroded) > 0


def per_image_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    class_names: list[str],
    boundary_tolerance: int,
) -> dict[str, float]:
    matrix = confusion_matrix_from_arrays(target, prediction, len(class_names))
    class_rows = class_metrics_from_confusion(matrix, class_names)
    aggregate = aggregate_confusion_metrics(matrix, class_rows)
    output = dict(aggregate)
    for row in class_rows:
        slug = metric_slug(str(row["class_name"]))
        for key in (
            "support",
            "predicted_support",
            "prevalence",
            "predicted_prevalence",
            "precision",
            "recall",
            "specificity",
            "dice",
            "iou",
            "relative_area_error",
            "absolute_relative_area_error",
        ):
            output[f"{slug}_{key}"] = float(row[key])
        if int(row["class_index"]) == 0:
            continue
        for key, value in boundary_metrics(
            target == int(row["class_index"]),
            prediction == int(row["class_index"]),
            tolerance=boundary_tolerance,
        ).items():
            output[f"{slug}_{key}"] = value
    return output


def aggregate_boundary_metrics(
    image_rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, float]:
    results: dict[str, float] = {}
    keys = (
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "surface_dice",
        "hausdorff_distance",
        "hausdorff_95",
        "average_symmetric_surface_distance",
    )
    for class_name in class_names[1:]:
        slug = metric_slug(class_name)
        for key in keys:
            values = [float(row.get(f"{slug}_{key}", float("nan"))) for row in image_rows]
            results[f"mean_{slug}_{key}"] = nanmean(values)
            finite = [value for value in values if np.isfinite(value)]
            results[f"median_{slug}_{key}"] = (
                float(np.median(finite)) if finite else float("nan")
            )
            results[f"valid_{slug}_{key}_count"] = float(len(finite))
    for key in keys:
        values = [
            value
            for class_name in class_names[1:]
            for value in [
                float(row.get(f"{metric_slug(class_name)}_{key}", float("nan")))
                for row in image_rows
            ]
        ]
        results[f"mean_foreground_{key}"] = nanmean(values)
    return results


def threshold_sweep(
    targets: np.ndarray,
    foreground_probabilities: np.ndarray,
    *,
    threshold_steps: int,
    configured_threshold: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    binary_targets = (targets.reshape(-1) > 0).astype(np.uint8)
    scores = foreground_probabilities.reshape(-1)
    thresholds = sorted(
        set(np.linspace(0.0, 1.0, threshold_steps).tolist() + [configured_threshold])
    )
    true_positive: list[int] = []
    false_positive: list[int] = []
    for threshold in thresholds:
        predicted = scores >= threshold
        true_positive.append(int(((binary_targets == 1) & predicted).sum()))
        false_positive.append(int(((binary_targets == 0) & predicted).sum()))
    return _threshold_rows_from_counts(
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(true_positive, dtype=np.int64),
        np.asarray(false_positive, dtype=np.int64),
        positive_count=int(binary_targets.sum()),
        negative_count=int((binary_targets == 0).sum()),
    )


def _threshold_rows_from_counts(
    thresholds: np.ndarray,
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    *,
    positive_count: int,
    negative_count: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, threshold in enumerate(thresholds):
        tp = float(true_positive[index])
        fp = float(false_positive[index])
        tn = float(negative_count - false_positive[index])
        fn = float(positive_count - true_positive[index])
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        npv = safe_divide(tn, tn + fn)
        dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        iou = safe_divide(tp, tp + fp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
                "balanced_accuracy": nanmean([recall, specificity]),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "npv": npv,
                "dice": dice,
                "f1": dice,
                "iou": iou,
                "false_positive_rate": safe_divide(fp, fp + tn),
                "false_negative_rate": safe_divide(fn, fn + tp),
            }
        )

    best_dice = max(rows, key=lambda row: (np.nan_to_num(row["dice"], nan=-1.0), -row["threshold"]))
    best_iou = max(rows, key=lambda row: (np.nan_to_num(row["iou"], nan=-1.0), -row["threshold"]))
    best_youden = max(
        rows,
        key=lambda row: (
            np.nan_to_num(row["recall"] + row["specificity"] - 1.0, nan=-1.0),
            -row["threshold"],
        ),
    )
    return rows, {
        "best_dice_threshold": best_dice["threshold"],
        "best_dice_at_threshold": best_dice["dice"],
        "best_iou_threshold": best_iou["threshold"],
        "best_iou_at_threshold": best_iou["iou"],
        "best_youden_threshold": best_youden["threshold"],
        "best_youden_j": best_youden["recall"] + best_youden["specificity"] - 1.0,
    }


@dataclass
class SegmentationResearchMetrics:
    metrics: dict[str, float]
    class_rows: list[dict[str, Any]]
    image_rows: list[dict[str, Any]]
    confusion_matrix: np.ndarray
    curves: dict[str, list[dict[str, float]]]
    threshold_rows: list[dict[str, float]]
