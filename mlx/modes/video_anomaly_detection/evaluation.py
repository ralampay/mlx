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
)
from mlx.modes.video_anomaly_detection.benchmark_artifacts import (
    VideoAnomalyBenchmarkArtifactWriter,
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
        artifact_writer: VideoAnomalyBenchmarkArtifactWriter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.checkpoint_loader = checkpoint_loader
        self.dataset_factory = dataset_factory
        self.artifact_writer = artifact_writer or VideoAnomalyBenchmarkArtifactWriter()

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
        metrics = {"clip_level": clip_metrics, "frame_level": frame_metrics}
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
        self.artifact_writer.write(
            output_dir,
            metrics=metrics,
            records=records,
            frame_records=frame_records,
            curves=curves,
            frame_curves=frame_curves,
            matrix=matrix,
            checkpoint=checkpoint,
            checkpoint_name=Path(config["model_path"]).name,
            dataset_name=Path(config["dataset_path"]).name,
            source_count=len({(row["ground_truth"], row["source"]) for row in records}),
            parameter_count=count_model_parameters(model),
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


__all__ = ["BenchmarkVideoAnomalyModel"]
