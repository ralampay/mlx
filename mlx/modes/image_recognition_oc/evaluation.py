from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlx.core.artifacts import sha256_file
from mlx.core.binary_metrics import compute_binary_score_metrics
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import count_model_parameters
from mlx.modes.image_recognition_oc.artifacts import (
    load_image_one_class_checkpoint,
    utc_timestamp,
)
from mlx.modes.image_recognition_oc.benchmark_artifacts import (
    ImageOneClassBenchmarkArtifactWriter,
)
from mlx.modes.image_recognition_oc.data import OneClassImageDataset
from mlx.modes.image_recognition_oc.requests import BenchmarkImageOneClassRequest


class BenchmarkImageOneClass:
    def __init__(
        self,
        request: BenchmarkImageOneClassRequest,
        *,
        reporter: WorkflowReporter | None = None,
        checkpoint_loader=load_image_one_class_checkpoint,
        dataset_factory=OneClassImageDataset,
        artifact_writer: ImageOneClassBenchmarkArtifactWriter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.checkpoint_loader = checkpoint_loader
        self.dataset_factory = dataset_factory
        self.artifact_writer = artifact_writer or ImageOneClassBenchmarkArtifactWriter()

    def execute(self) -> dict[str, Any]:
        config = self.request.to_config()
        if not config.get("model_path"):
            raise MLXUserError("Benchmarking requires --model-path.")
        if not config.get("output_path"):
            raise MLXUserError("Benchmarking requires --output.")
        if int(config.get("batch_size", 0)) < 1:
            raise MLXUserError("--batch-size must be at least 1 for benchmarking.")
        if int(config.get("workers", 0)) < 0:
            raise MLXUserError("--workers must be greater than or equal to zero.")
        device = str(config["device"])
        model, checkpoint, stored, algorithm = self.checkpoint_loader(
            config["model_path"],
            device=device,
            model_name=config.get("model"),
            backbone_name=config.get("backbone"),
        )
        threshold = checkpoint.get("threshold")
        if threshold is None or not np.isfinite(float(threshold)):
            raise MLXUserError(
                "Benchmark checkpoint has no calibrated threshold; benchmark never calibrates on test data."
            )
        dataset = self.dataset_factory(
            config["dataset_path"],
            split="test",
            height=int(stored["height"]),
            width=int(stored["width"]),
            colored=bool(stored["colored"]),
            normal_only=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("workers", 0)),
        )
        records = self._predict(algorithm, model, loader, device, float(threshold), config["dataset_path"])
        metrics, curves, matrix = compute_binary_score_metrics(
            [row["ground_truth"] for row in records],
            [row["anomaly_score"] for row in records],
            threshold=float(threshold),
            task_label="One-class image benchmark",
        )
        output_dir = Path(str(config["output_path"])).expanduser()
        metadata = {
            "mode": "image_recognition_oc",
            "evaluated_at": utc_timestamp(),
            "checkpoint": Path(str(config["model_path"])).name,
            "checkpoint_sha256": sha256_file(config["model_path"]),
            "model": checkpoint["model_name"],
            "backbone": checkpoint["backbone_name"],
            "dataset": str(config["dataset_path"]),
            "height": stored["height"],
            "width": stored["width"],
            "colored": stored["colored"],
            "device": device,
            "batch_size": int(config["batch_size"]),
            "threshold": float(threshold),
            "score_type": checkpoint["score_type"],
            "evaluator_version": 1,
        }
        self.artifact_writer.write(
            output_dir,
            metrics=metrics,
            records=records,
            curves=curves,
            matrix=matrix,
            metadata=metadata,
            checkpoint=checkpoint,
            parameter_count=count_model_parameters(model),
            plots=bool(config.get("plots", True)),
        )
        emit(
            self.reporter,
            "success",
            f"Benchmark complete: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}",
            payload={"event": "image_one_class_benchmark", "metrics": metrics, "output": str(output_dir)},
        )
        return {
            "metrics": metrics,
            "predictions": records,
            "output_dir": output_dir,
            "confusion_matrix": matrix,
        }

    @staticmethod
    @torch.no_grad()
    def _predict(algorithm, model, loader, device: str, threshold: float, dataset_path):
        model.eval()
        root = Path(str(dataset_path)).expanduser()
        records = []
        for images, labels, paths in loader:
            scores = algorithm.scores(model, images.to(device)).detach().cpu().tolist()
            for label, score, path in zip(labels.tolist(), scores, paths, strict=True):
                if not np.isfinite(float(score)):
                    raise MLXUserError("One-class image benchmark produced a non-finite score.")
                image_path = Path(path)
                try:
                    relative = str(image_path.relative_to(root))
                except ValueError:
                    relative = str(image_path)
                records.append(
                    {
                        "image": relative,
                        "ground_truth": int(label),
                        "ground_truth_label": "anomaly" if int(label) else "normal",
                        "anomaly_score": float(score),
                        "threshold": threshold,
                        "predicted_anomaly": int(float(score) > threshold),
                        "predicted_label": "anomaly" if float(score) > threshold else "normal",
                    }
                )
        return records


__all__ = ["BenchmarkImageOneClass"]
