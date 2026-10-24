"""Classification benchmark artifact rendering, separate from scoring and inference."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.modes.image_classification.evaluation_records import OneShotPredictionRecord

def _write_benchmark_artifacts(
    output_dir: Path,
    *,
    results: dict[str, float],
    targets: list[int],
    preds: list[int],
    probabilities: np.ndarray | None,
    class_names: list[str] | None,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(output_dir / "metrics.csv", results)
    _write_confusion_matrix_artifacts(
        output_dir,
        targets=targets,
        preds=preds,
        class_names=class_names,
    )
    _write_roc_curve_artifact(
        output_dir / "roc_curve.png",
        targets=targets,
        probabilities=probabilities,
        class_names=class_names,
        reporter=reporter,
    )
    emit(reporter, "success", f"Benchmark artifacts written to {output_dir}")


def _write_metrics_csv(csv_path: Path, results: dict[str, float]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in sorted(results.items()):
            writer.writerow([metric_name, f"{metric_value:.6f}"])


def _write_confusion_matrix_artifacts(
    output_dir: Path,
    *,
    targets: list[int],
    preds: list[int],
    class_names: list[str] | None,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(range(len(class_names))) if class_names else sorted(set(targets) | set(preds))
    matrix = confusion_matrix(targets, preds, labels=labels)
    resolved_names = class_names or [str(label) for label in labels]

    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual/predicted", *resolved_names])
        for class_name, row in zip(resolved_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    figure_width = max(6, min(16, len(resolved_names) * 1.2))
    figure_height = max(5, min(14, len(resolved_names) * 1.0))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(resolved_names)),
        yticks=np.arange(len(resolved_names)),
        xticklabels=resolved_names,
        yticklabels=resolved_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_roc_curve_artifact(
    image_path: Path,
    *,
    targets: list[int],
    probabilities: np.ndarray | None,
    class_names: list[str] | None,
    reporter: WorkflowReporter | None = None,
) -> None:
    import matplotlib.pyplot as plt

    reporter = reporter or NullWorkflowReporter()
    if probabilities is None or len(targets) == 0:
        return

    target_array = np.asarray(targets, dtype=np.int64)
    unique_targets = np.unique(target_array)
    if unique_targets.size < 2:
        return

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if probabilities.ndim == 1 or (probabilities.ndim == 2 and probabilities.shape[1] == 1):
            positive_scores = probabilities.reshape(-1)
            fpr, tpr, _ = roc_curve(target_array, positive_scores)
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}", linewidth=2)
        elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
            positive_scores = probabilities[:, 1]
            fpr, tpr, _ = roc_curve(target_array, positive_scores)
            positive_label = class_names[1] if class_names and len(class_names) > 1 else "positive"
            ax.plot(fpr, tpr, label=f"{positive_label} (AUC={auc(fpr, tpr):.4f})", linewidth=2)
        else:
            class_count = probabilities.shape[1]
            classes = np.arange(class_count)
            binarized_targets = label_binarize(target_array, classes=classes)
            for class_index in range(class_count):
                if binarized_targets[:, class_index].max() == 0:
                    continue
                fpr, tpr, _ = roc_curve(binarized_targets[:, class_index], probabilities[:, class_index])
                label = class_names[class_index] if class_names and class_index < len(class_names) else f"class_{class_index}"
                ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.4f})", linewidth=2)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set(
            title="ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.05),
        )
        ax.legend(loc="lower right", fontsize="small")
        fig.tight_layout()
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ValueError as exc:
        emit(reporter, "warning", f"ROC curve export skipped: {exc}")


def _write_one_shot_research_artifacts(
    output_dir: Path,
    *,
    prediction_records: list[OneShotPredictionRecord],
    threshold_rows: list[dict[str, float]],
    targets: list[int],
    probabilities: np.ndarray,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_one_shot_pair_predictions(output_dir / "pair_predictions.csv", prediction_records)
    _write_one_shot_threshold_metrics(output_dir / "threshold_metrics.csv", threshold_rows)
    _write_precision_recall_curve_artifact(
        output_dir / "precision_recall_curve.png",
        targets=targets,
        probabilities=probabilities,
    )
    _write_score_distribution_artifact(
        output_dir / "score_distribution.png",
        targets=targets,
        probabilities=probabilities,
    )
    emit(reporter, "success", f"One-shot research artifacts written to {output_dir}")


def _write_one_shot_pair_predictions(
    csv_path: Path,
    prediction_records: list[OneShotPredictionRecord],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "pair_id",
                "image_1",
                "label_1",
                "image_2",
                "label_2",
                "target",
                "predicted",
                "same_probability",
            ]
        )
        for index, record in enumerate(prediction_records, start=1):
            writer.writerow(
                [
                    index,
                    str(record.image_one),
                    record.label_one,
                    str(record.image_two),
                    record.label_two,
                    record.target,
                    record.predicted,
                    f"{record.same_probability:.8f}",
                ]
            )


def _write_one_shot_threshold_metrics(
    csv_path: Path,
    threshold_rows: list[dict[str, float]],
) -> None:
    fieldnames = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in threshold_rows:
            writer.writerow(
                {
                    key: f"{row[key]:.8f}" if key in row else ""
                    for key in fieldnames
                }
            )


def _write_n_way_predictions(csv_path: Path, prediction_rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_image",
        "query_label",
        "predicted_label",
        "best_reference_image",
        "best_same_probability",
        "correct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow(
                {
                    **row,
                    "best_same_probability": f"{row['best_same_probability']:.8f}",
                }
            )


def _write_precision_recall_curve_artifact(
    image_path: Path,
    *,
    targets: list[int],
    probabilities: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if len(targets) == 0 or np.unique(np.asarray(targets, dtype=np.int64)).size < 2:
        return

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    average_precision = average_precision_score(targets, probabilities)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"AP={average_precision:.4f}", linewidth=2)
    ax.set(
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.05),
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_score_distribution_artifact(
    image_path: Path,
    *,
    targets: list[int],
    probabilities: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if len(targets) == 0:
        return

    target_array = np.asarray(targets, dtype=np.int64)
    same_scores = probabilities[target_array == 1]
    different_scores = probabilities[target_array == 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0.0, 1.0, 41)
    if different_scores.size:
        ax.hist(different_scores, bins=bins, alpha=0.65, label="different", density=True)
    if same_scores.size:
        ax.hist(same_scores, bins=bins, alpha=0.65, label="same", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set(
        title="One-Shot Similarity Score Distribution",
        xlabel="Predicted same-class probability",
        ylabel="Density",
        xlim=(0.0, 1.0),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

class ClassificationBenchmarkArtifacts:
    """Default CSV/plot writer; all benchmark output passes through this port."""

    write_metrics = staticmethod(_write_benchmark_artifacts)
    write_one_shot = staticmethod(_write_one_shot_research_artifacts)
    write_n_way = staticmethod(_write_n_way_predictions)
