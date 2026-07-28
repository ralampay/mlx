from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    write_csv(
        path,
        [{"metric": key, "value": value} for key, value in sorted(metrics.items())],
        ["metric", "value"],
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_value(value), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def write_confusion_matrix_artifacts(
    output_dir: Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    rows = []
    for class_name, values in zip(class_names, matrix, strict=True):
        rows.append(
            {
                "actual/predicted": class_name,
                **{name: int(value) for name, value in zip(class_names, values, strict=True)},
            }
        )
    write_csv(
        output_dir / "confusion_matrix.csv",
        rows,
        ["actual/predicted", *class_names],
    )
    _plot_confusion_matrix(
        output_dir / "confusion_matrix.png",
        matrix.astype(np.float64),
        class_names,
        title="Pixel Confusion Matrix",
        value_format=".0f",
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    _plot_confusion_matrix(
        output_dir / "confusion_matrix_normalized.png",
        normalized,
        class_names,
        title="Row-Normalized Pixel Confusion Matrix",
        value_format=".3f",
    )


def write_curve_artifacts(
    output_dir: Path,
    curves: dict[str, list[dict[str, float]]],
    class_names: list[str],
) -> None:
    roc_rows = curves.get("roc", [])
    pr_rows = curves.get("precision_recall", [])
    calibration_rows = curves.get("calibration", [])
    write_csv(output_dir / "roc_curves.csv", roc_rows)
    write_csv(output_dir / "precision_recall_curves.csv", pr_rows)
    write_csv(output_dir / "calibration_curve.csv", calibration_rows)

    if roc_rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        for class_index, class_name in enumerate(class_names):
            rows = [row for row in roc_rows if int(row["class_index"]) == class_index]
            if rows:
                ax.plot(
                    [row["false_positive_rate"] for row in rows],
                    [row["true_positive_rate"] for row in rows],
                    label=class_name,
                )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set(title="Pixel ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curves.png", dpi=200)
        plt.close(fig)

    if pr_rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        for class_index, class_name in enumerate(class_names):
            rows = [row for row in pr_rows if int(row["class_index"]) == class_index]
            if rows:
                ax.plot(
                    [row["recall"] for row in rows],
                    [row["precision"] for row in rows],
                    label=class_name,
                )
        ax.set(
            title="Pixel Precision-Recall Curves",
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1.05),
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "precision_recall_curves.png", dpi=200)
        plt.close(fig)

    if calibration_rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            [row["confidence"] for row in calibration_rows],
            [row["accuracy"] for row in calibration_rows],
            marker="o",
            label="model",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
        ax.set(
            title="Reliability Diagram",
            xlabel="Mean Confidence",
            ylabel="Observed Accuracy",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "calibration_curve.png", dpi=200)
        plt.close(fig)


def write_class_metrics_plot(output_dir: Path, class_rows: list[dict[str, Any]]) -> None:
    metrics = ("precision", "recall", "specificity", "dice", "iou")
    names = [str(row["class_name"]) for row in class_rows]
    x_positions = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
    for metric_index, metric in enumerate(metrics):
        ax.bar(
            x_positions + metric_index * width,
            [float(row[metric]) for row in class_rows],
            width,
            label=metric,
        )
    ax.set(
        title="Per-Class Segmentation Metrics",
        ylabel="Score",
        ylim=(0, 1.05),
        xticks=x_positions + width * (len(metrics) - 1) / 2,
        xticklabels=names,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=200)
    plt.close(fig)


def write_image_metric_distribution(
    output_dir: Path,
    image_rows: list[dict[str, Any]],
) -> None:
    metric_names = ("pixel_accuracy", "mean_foreground_dice", "mean_foreground_iou")
    values = [
        [
            float(row[metric])
            for row in image_rows
            if metric in row and np.isfinite(float(row[metric]))
        ]
        for metric in metric_names
    ]
    if not any(values):
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(values, tick_labels=metric_names, showmeans=True)
    ax.set(title="Per-Image Metric Distributions", ylabel="Score", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output_dir / "per_image_metric_distributions.png", dpi=200)
    plt.close(fig)


def write_threshold_artifacts(
    output_dir: Path,
    threshold_rows: list[dict[str, float]],
) -> None:
    if not threshold_rows:
        return
    write_csv(output_dir / "threshold_metrics.csv", threshold_rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    for metric in ("dice", "iou", "precision", "recall", "specificity"):
        ax.plot(
            [row["threshold"] for row in threshold_rows],
            [row[metric] for row in threshold_rows],
            label=metric,
        )
    ax.set(
        title="Binary Segmentation Threshold Analysis",
        xlabel="Foreground Probability Threshold",
        ylabel="Score",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_curves.png", dpi=200)
    plt.close(fig)


def write_training_curves(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train loss")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="validation loss")
    axes[0].set(title="Segmentation Loss", ylabel="Loss")
    axes[0].legend()
    for metric in (
        "pixel_accuracy",
        "macro_dice",
        "mean_foreground_dice",
        "mean_iou",
        "mean_foreground_iou",
    ):
        if metric in history[0]:
            axes[1].plot(epochs, [float(row[metric]) for row in history], label=metric)
    axes[1].set(title="Validation Metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1.05))
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    class_names: list[str],
    *,
    title: str,
    value_format: str,
) -> None:
    figure_size = max(6, min(16, len(class_names) * 1.2))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        if np.isposinf(numeric):
            return "inf"
        if np.isneginf(numeric):
            return "-inf"
        return f"{numeric:.8f}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value
