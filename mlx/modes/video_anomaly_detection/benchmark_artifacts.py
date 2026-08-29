from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.modes.video_anomaly_detection.artifacts import write_csv, write_json, write_jsonl


class VideoAnomalyBenchmarkArtifactWriter:
    """Own the deterministic machine-readable and research presentation artifacts."""

    def write(
        self,
        output_dir: Path,
        *,
        metrics: dict[str, Any],
        records: list[dict[str, Any]],
        frame_records: list[dict[str, Any]],
        curves: dict[str, Any],
        frame_curves: dict[str, Any],
        matrix,
        checkpoint: dict[str, Any],
        checkpoint_name: str,
        dataset_name: str,
        source_count: int,
        parameter_count: int,
        metadata: dict[str, Any],
        plots: bool,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", metrics)
        write_csv(
            output_dir / "metrics.csv",
            [
                {"level": level, "metric": key, "value": value}
                for level, values in metrics.items()
                for key, value in sorted(values.items())
            ],
            ["level", "metric", "value"],
        )
        public_records = [
            {key: value for key, value in row.items() if key != "frame_indices"}
            for row in records
        ]
        write_csv(output_dir / "predictions.csv", public_records)
        write_jsonl(output_dir / "predictions.jsonl", records)
        write_csv(output_dir / "frame_predictions.csv", frame_records)
        write_csv(output_dir / "roc_curve.csv", curves["roc"])
        write_csv(output_dir / "pr_curve.csv", curves["precision_recall"])
        write_csv(output_dir / "frame_roc_curve.csv", frame_curves["roc"])
        write_csv(output_dir / "frame_pr_curve.csv", frame_curves["precision_recall"])
        write_json(output_dir / "run_metadata.json", metadata)
        if plots:
            _write_benchmark_plots(output_dir, records, curves, matrix)
        _write_report(
            output_dir / "benchmark_report.md",
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            dataset_name=dataset_name,
            source_count=source_count,
            parameter_count=parameter_count,
            clip_metrics=metrics["clip_level"],
            frame_metrics=metrics["frame_level"],
            metadata=metadata,
            plots=plots,
        )


def _write_benchmark_plots(output_dir: Path, records, curves, matrix) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [row["false_positive_rate"] for row in curves["roc"]],
        [row["true_positive_rate"] for row in curves["roc"]],
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="Video Anomaly ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [row["recall"] for row in curves["precision_recall"]],
        [row["precision"] for row in curves["precision_recall"]],
    )
    ax.set(title="Video Anomaly Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name in ((0, "normal"), (1, "anomaly")):
        ax.hist(
            [row["anomaly_score"] for row in records if row["ground_truth"] == label],
            bins=20,
            alpha=0.55,
            label=name,
        )
    ax.axvline(records[0]["threshold"], color="black", linestyle="--", label="threshold")
    ax.set(title="Window Score Distribution", xlabel="Squared distance", ylabel="Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["normal", "anomaly"],
        yticklabels=["normal", "anomaly"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Window Confusion Matrix",
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    checkpoint,
    checkpoint_name,
    dataset_name,
    source_count,
    parameter_count,
    clip_metrics,
    frame_metrics,
    metadata,
    plots,
) -> None:
    def table(values):
        rows = ["| Metric | Value |", "| --- | ---: |"]
        rows.extend(f"| {key} | {_format_metric(value)} |" for key, value in sorted(values.items()))
        return "\n".join(rows)

    artifacts = ["metrics.json", "metrics.csv", "predictions.csv", "frame_predictions.csv"]
    if plots:
        artifacts.extend(["roc_curve.png", "pr_curve.png", "score_distribution.png", "confusion_matrix.png"])
    temporal_description = (
        f"3D inflation (kernel {checkpoint['backbone_temporal_kernel_size']}, temporal stride 1)"
        if checkpoint.get("backbone_mode") == "3d"
        else checkpoint.get("temporal_model", "tcn")
    )
    text = f"""# Video Anomaly Detection Benchmark

## Model

- Backbone: `{checkpoint['model_name']}`
- Backbone class: `{checkpoint.get('backbone_class', 'legacy image feature adapter')}`
- Backbone mode: `{checkpoint.get('backbone_mode', 'frame-2d')}`
- Temporal modeling: `{temporal_description}`
- Pretrained provenance: `{checkpoint.get('pretrained_provenance', 'not recorded')}`
- SVDD dimensions: hidden `{checkpoint['svdd_hidden_dim']}`, embedding `{checkpoint['svdd_dim']}`
- Checkpoint: `{checkpoint_name}`
- Parameter count: {parameter_count}

## Dataset

- Dataset: `{dataset_name}`
- Source sequences: {source_count}
- Normal windows: {clip_metrics['normal_samples']}
- Anomaly windows: {clip_metrics['anomaly_samples']}
- Clip length: {checkpoint['clip_length']}
- Frame stride: {checkpoint['frame_stride']}
- Image dimensions: {checkpoint['input_height']}x{checkpoint['input_width']}

## Threshold

- Calibration quantile: {checkpoint['svdd_quantile']}
- Stored threshold: {checkpoint['svdd_threshold']}
- Score: squared Euclidean distance from the fixed normal center; higher is more anomalous.

## Clip-Level Results

{table(clip_metrics)}

## Frame-Level Results

Aggregation: `{metadata['frame_aggregation']}` over overlapping window scores.

{table(frame_metrics)}

## Score Statistics

- Normal: mean {clip_metrics['normal_score_mean']:.6f}, std {clip_metrics['normal_score_std']:.6f}
- Anomaly: mean {clip_metrics['anomaly_score_mean']:.6f}, std {clip_metrics['anomaly_score_std']:.6f}

## Artifacts

""" + "\n".join(f"- [{name}]({name})" for name in artifacts) + f"""

## Reproducibility

- Device: `{metadata['device']}`
- Batch size: {metadata['batch_size']}
- Checkpoint SHA-256: `{metadata['checkpoint_sha256']}`
"""
    path.write_text(text, encoding="utf-8")


def _format_metric(value) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


__all__ = ["VideoAnomalyBenchmarkArtifactWriter"]
