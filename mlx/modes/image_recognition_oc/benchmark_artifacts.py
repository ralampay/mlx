from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.artifacts import write_csv, write_json_atomic


class ImageOneClassBenchmarkArtifactWriter:
    def write(
        self,
        output_dir: Path,
        *,
        metrics: dict[str, Any],
        records: list[dict[str, Any]],
        curves: dict[str, Any],
        matrix,
        metadata: dict[str, Any],
        checkpoint: dict[str, Any],
        parameter_count: int,
        plots: bool,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir / "metrics.json", metrics)
        write_csv(
            output_dir / "metrics.csv",
            ({"metric": key, "value": value} for key, value in sorted(metrics.items())),
            fieldnames=("metric", "value"),
        )
        write_csv(output_dir / "predictions.csv", records)
        _write_jsonl(output_dir / "predictions.jsonl", records)
        write_csv(output_dir / "roc_curve.csv", curves["roc"])
        write_csv(output_dir / "pr_curve.csv", curves["precision_recall"])
        write_json_atomic(output_dir / "run_metadata.json", metadata)
        if plots:
            _write_plots(output_dir, records, curves, matrix)
        _write_report(
            output_dir / "benchmark_report.md",
            metrics=metrics,
            metadata=metadata,
            checkpoint=checkpoint,
            parameter_count=parameter_count,
            plots=plots,
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    from mlx.core.artifacts import json_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    except OSError as exc:
        from mlx.core.exceptions import MLXUserError

        raise MLXUserError(f"Unable to write JSONL artifact '{path}': {exc}") from exc


def _write_plots(output_dir: Path, records, curves, matrix) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        [row["false_positive_rate"] for row in curves["roc"]],
        [row["true_positive_rate"] for row in curves["roc"]],
    )
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(title="One-Class Image ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        [row["recall"] for row in curves["precision_recall"]],
        [row["precision"] for row in curves["precision_recall"]],
    )
    axis.set(title="One-Class Image Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    figure.tight_layout()
    figure.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for label, name in ((0, "normal"), (1, "anomaly")):
        axis.hist(
            [row["anomaly_score"] for row in records if row["ground_truth"] == label],
            bins=20,
            alpha=0.55,
            label=name,
        )
    axis.axvline(records[0]["threshold"], color="black", linestyle="--", label="threshold")
    axis.set(title="Image Score Distribution", xlabel="Anomaly score", ylabel="Count")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "score_distribution.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["normal", "anomaly"],
        yticklabels=["normal", "anomaly"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Image Confusion Matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def _write_report(path, *, metrics, metadata, checkpoint, parameter_count, plots) -> None:
    rows = ["| Metric | Value |", "| --- | ---: |"]
    rows.extend(
        f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |"
        for key, value in sorted(metrics.items())
    )
    artifacts = [
        "metrics.json",
        "metrics.csv",
        "predictions.csv",
        "predictions.jsonl",
        "roc_curve.csv",
        "pr_curve.csv",
        "run_metadata.json",
    ]
    if plots:
        artifacts.extend(("roc_curve.png", "pr_curve.png", "score_distribution.png", "confusion_matrix.png"))
    text = f"""# One-Class Image Recognition Benchmark

## Model

- One-class model: `{checkpoint['model_name']}`
- Backbone: `{checkpoint['backbone_name']}`
- Backbone class: `{checkpoint['backbone_class']}`
- Parameters: {parameter_count}
- Checkpoint: `{metadata['checkpoint']}`

## Dataset and threshold

- Dataset: `{metadata['dataset']}`
- Normal images: {metrics['normal_samples']}
- Anomaly images: {metrics['anomaly_samples']}
- Calibration quantile: {checkpoint['svdd_quantile']}
- Stored threshold: {checkpoint['threshold']}
- Score: squared Euclidean distance from the fixed normal center; higher is more anomalous.

## Results

{chr(10).join(rows)}

## Artifacts

{chr(10).join(f'- [{name}]({name})' for name in artifacts)}

## Reproducibility

- Device: `{metadata['device']}`
- Batch size: {metadata['batch_size']}
- Checkpoint SHA-256: `{metadata['checkpoint_sha256']}`
"""
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        from mlx.core.exceptions import MLXUserError

        raise MLXUserError(f"Unable to write benchmark report '{path}': {exc}") from exc


__all__ = ["ImageOneClassBenchmarkArtifactWriter"]
