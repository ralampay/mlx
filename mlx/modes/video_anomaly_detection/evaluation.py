from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import count_model_parameters
from mlx.modes.video_anomaly_detection.artifacts import (
    load_video_anomaly_checkpoint,
    sha256_file,
    utc_timestamp,
    write_csv,
    write_json,
    write_jsonl,
)
from mlx.modes.video_anomaly_detection.clips import aggregate_frame_scores
from mlx.modes.video_anomaly_detection.data import VideoClipDataset, collate_clip_samples
from mlx.modes.video_anomaly_detection.metrics import compute_binary_metrics
from mlx.modes.video_anomaly_detection.requests import BenchmarkVideoAnomalyRequest


class BenchmarkVideoAnomalyModel:
    def __init__(
        self,
        request: BenchmarkVideoAnomalyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        checkpoint_loader=load_video_anomaly_checkpoint,
        dataset_factory=VideoClipDataset,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.checkpoint_loader = checkpoint_loader
        self.dataset_factory = dataset_factory

    def execute(self) -> dict[str, Any]:
        config = self.request.to_config()
        if not config.get("model_path"):
            raise MLXUserError("Benchmarking requires --model-path.")
        if not config.get("output_path"):
            raise MLXUserError("Benchmarking requires --output.")
        aggregation = str(config.get("frame_aggregation", "mean"))
        if aggregation not in {"mean", "max"}:
            raise MLXUserError("--frame-aggregation must be mean or max.")
        model, checkpoint, stored = self.checkpoint_loader(
            config["model_path"],
            device=str(config["device"]),
            model_name=config.get("model"),
        )
        threshold = checkpoint.get("svdd_threshold")
        if threshold is None or not np.isfinite(float(threshold)):
            raise MLXUserError(
                "Benchmark checkpoint has no calibrated threshold; benchmark never calibrates on test data."
            )
        dataset = self.dataset_factory(
            config["dataset_path"],
            split="test",
            clip_length=stored["clip_length"],
            frame_stride=stored["frame_stride"],
            height=stored["height"],
            width=stored["width"],
            normal_only=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("workers", 0)),
            collate_fn=collate_clip_samples,
        )
        records = self._predict(model, loader, str(config["device"]), float(threshold))
        labels = [int(row["ground_truth"]) for row in records]
        scores = [float(row["anomaly_score"]) for row in records]
        clip_metrics, curves, matrix = compute_binary_metrics(
            labels, scores, threshold=float(threshold)
        )
        frame_records = aggregate_frame_scores(records, method=aggregation)
        frame_metrics, frame_curves, frame_matrix = compute_binary_metrics(
            [row["ground_truth"] for row in frame_records],
            [row["anomaly_score"] for row in frame_records],
            threshold=float(threshold),
        )
        output_dir = Path(config["output_path"]).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"clip_level": clip_metrics, "frame_level": frame_metrics}
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
        metadata = {
            "mode": "video_anomaly_detection",
            "evaluated_at": utc_timestamp(),
            "checkpoint": Path(config["model_path"]).name,
            "checkpoint_sha256": sha256_file(config["model_path"]),
            "model": checkpoint["model_name"],
            "backbone_mode": checkpoint.get("backbone_mode", "frame-2d"),
            "backbone_class": checkpoint.get("backbone_class"),
            "temporal_model": checkpoint.get("temporal_model"),
            "backbone_temporal_kernel_size": checkpoint.get(
                "backbone_temporal_kernel_size"
            ),
            "dataset": str(config["dataset_path"]),
            "clip_length": stored["clip_length"],
            "frame_stride": stored["frame_stride"],
            "height": stored["height"],
            "width": stored["width"],
            "device": config["device"],
            "batch_size": int(config["batch_size"]),
            "threshold": float(threshold),
            "score_type": checkpoint["score_type"],
            "drax_fusion_mode": checkpoint.get("drax_fusion_mode"),
            "pretrained_provenance": checkpoint.get("pretrained_provenance"),
            "checkpoint_version": checkpoint.get("checkpoint_version", 1),
            "frame_aggregation": aggregation,
            "evaluator_version": 1,
        }
        write_json(output_dir / "run_metadata.json", metadata)
        if bool(config.get("plots", True)):
            _write_benchmark_plots(output_dir, records, curves, matrix)
        _write_report(
            output_dir / "benchmark_report.md",
            checkpoint=checkpoint,
            checkpoint_name=Path(config["model_path"]).name,
            dataset_name=Path(config["dataset_path"]).name,
            source_count=len({(row["ground_truth"], row["source"]) for row in records}),
            parameter_count=count_model_parameters(model),
            clip_metrics=clip_metrics,
            frame_metrics=frame_metrics,
            metadata=metadata,
            plots=bool(config.get("plots", True)),
        )
        emit(
            self.reporter,
            "success",
            f"Benchmark complete: clip AUROC={clip_metrics['auroc']:.4f}, AUPRC={clip_metrics['auprc']:.4f}",
            payload={"event": "video_anomaly_benchmark", "metrics": metrics, "output": str(output_dir)},
        )
        return {
            "metrics": metrics,
            "predictions": records,
            "frame_predictions": frame_records,
            "output_dir": output_dir,
            "confusion_matrix": matrix,
            "frame_confusion_matrix": frame_matrix,
        }

    @staticmethod
    @torch.no_grad()
    def _predict(model, loader, device: str, threshold: float):
        model.eval()
        records = []
        for clips, labels, metadata in loader:
            scores = model(clips.to(device)).anomaly_score.cpu().tolist()
            for label, score, item in zip(labels.tolist(), scores, metadata, strict=True):
                records.append(
                    {
                        **item,
                        "ground_truth": int(label),
                        "anomaly_score": float(score),
                        "threshold": threshold,
                        "predicted_anomaly": int(score > threshold),
                    }
                )
        return records


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
        rows.extend(
            f"| {key} | {_format_metric(value)} |"
            for key, value in sorted(values.items())
        )
        return "\n".join(rows)

    artifacts = ["metrics.json", "metrics.csv", "predictions.csv", "frame_predictions.csv"]
    if plots:
        artifacts.extend(
            [
                "roc_curve.png",
                "pr_curve.png",
                "score_distribution.png",
                "confusion_matrix.png",
            ]
        )
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


__all__ = ["BenchmarkVideoAnomalyModel"]
