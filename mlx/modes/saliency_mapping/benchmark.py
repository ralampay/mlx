from __future__ import annotations

import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import count_model_parameters
from mlx.modes.saliency_mapping.checkpoints import load_checkpoint_bundle
from mlx.modes.saliency_mapping.data import SaliencyDataset
from mlx.modes.saliency_mapping.losses import SaliencyHybridLoss
from mlx.modes.saliency_mapping.metrics import (
    SaliencyMetricAccumulator,
    per_image_saliency_metrics,
)
from mlx.modes.saliency_mapping.models import build_saliency_model, grouped_model_names
from mlx.modes.saliency_mapping.requests import BenchmarkSaliencyRequest
from mlx.modes.saliency_mapping.visualization import (
    compose_saliency_panel,
    probability_to_gray,
    probability_to_heatmap,
    saliency_overlay,
    tensor_to_rgb,
    write_rgb,
)


class BenchmarkSaliencyMapping:
    def __init__(
        self,
        request: BenchmarkSaliencyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        model_registry=None,
    ) -> None:
        self.model_registry = model_registry
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> dict[str, float]:
        if self.request.threshold_steps < 2:
            raise MLXUserError("--threshold-steps must be at least 2.")
        config = self.request.to_config()
        if self.request.model_path:
            model, metadata = load_checkpoint_bundle(config, **({"model_registry": self.model_registry} if self.model_registry is not None else {}))
        else:
            model_name = str(self.request.model)
            model = build_saliency_model(model_name, config, **({"registry": self.model_registry} if self.model_registry is not None else {}))
            metadata = {
                "checkpoint_path": None,
                "model_name": model_name,
                "input_size": self.request.input_size,
                "transform": self.request.transform,
                "colored": self.request.colored,
                "output_channels": 1,
            }
        device = self.request.device
        model = model.to(device).eval()
        dataset = SaliencyDataset(
            self.request.dataset_path,
            split=self.request.split,
            input_size=tuple(metadata["input_size"]),
            colored=bool(metadata["colored"]),
            transform=str(metadata["transform"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.request.batch_size),
            shuffle=False,
            num_workers=max(0, self.request.workers),
        )
        output_dir = Path(self.request.output_path).expanduser() if self.request.output_path else None
        metrics, image_rows, threshold_rows, timing = self._evaluate(
            model, dataset, loader, output_dir
        )
        metrics.update(timing)
        metrics["parameter_count"] = float(count_model_parameters(model))
        if output_dir is not None:
            self._write_artifacts(output_dir, metrics, image_rows, threshold_rows, metadata)
        emit(
            self.reporter,
            "success",
            "Saliency benchmark complete.",
            payload={"event": "saliency_benchmark", "metrics": metrics},
        )
        return metrics

    def _evaluate(self, model, dataset, loader, output_dir):
        accumulator = SaliencyMetricAccumulator(self.request.threshold_steps)
        criterion = SaliencyHybridLoss(
            bce_weight=self.request.bce_weight,
            ssim_weight=self.request.ssim_weight,
            iou_weight=self.request.iou_weight,
        )
        totals = {name: 0.0 for name in ("loss", "bce_loss", "ssim_loss", "iou_loss")}
        image_rows = []
        batch_times = []
        wall_start = time.perf_counter()
        sample_offset = 0
        if self.request.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch_index, (images, targets) in enumerate(loader, start=1):
                device_images = images.to(self.request.device)
                device_targets = targets.to(self.request.device)
                self._synchronize()
                start = time.perf_counter()
                logits = model(device_images)
                self._synchronize()
                elapsed = time.perf_counter() - start
                batch_times.append(elapsed)
                loss_result = criterion(logits, device_targets).as_dict()
                probabilities = torch.sigmoid(logits).cpu().numpy()
                target_values = targets.numpy()
                accumulator.update(probabilities, target_values)
                for key, value in loss_result.items():
                    totals[key] += float(value.item()) * len(images)
                for index in range(len(images)):
                    image_path, target_path = dataset.samples[sample_offset + index]
                    values, _ = per_image_saliency_metrics(
                        probabilities[index], target_values[index], threshold_steps=self.request.threshold_steps
                    )
                    row = {
                        "image": str(image_path),
                        "mask": str(target_path),
                        "width": int(probabilities.shape[-1]),
                        "height": int(probabilities.shape[-2]),
                        "inference_ms": elapsed * 1000.0 / len(images),
                        **values,
                    }
                    image_rows.append(row)
                    if output_dir is not None and self.request.save_images:
                        self._write_prediction(
                            output_dir,
                            image_path.stem,
                            images[index],
                            target_values[index, 0],
                            probabilities[index, 0],
                            sample_offset + index,
                        )
                sample_offset += len(images)
                emit(
                    self.reporter,
                    "progress",
                    f"Benchmarked saliency batch {batch_index} of {len(loader)}.",
                    current=batch_index,
                    total=len(loader),
                )
        metrics, threshold_rows = accumulator.finalize()
        for key, value in totals.items():
            metrics[key] = value / max(1, len(dataset))
        wall = time.perf_counter() - wall_start
        forward = sum(batch_times)
        per_image = np.asarray([row["inference_ms"] for row in image_rows], dtype=np.float64)
        per_batch = np.asarray(batch_times, dtype=np.float64) * 1000.0
        timing = {
            "wall_time_seconds": wall,
            "forward_time_seconds": forward,
            "images_per_second_wall": len(dataset) / wall if wall else float("nan"),
            "images_per_second_forward": len(dataset) / forward if forward else float("nan"),
            "mean_batch_latency_ms": float(per_batch.mean()),
            "median_batch_latency_ms": float(np.median(per_batch)),
            "p95_batch_latency_ms": float(np.percentile(per_batch, 95)),
            "mean_image_latency_ms": float(per_image.mean()),
            "median_image_latency_ms": float(np.median(per_image)),
            "p95_image_latency_ms": float(np.percentile(per_image, 95)),
        }
        if self.request.device.startswith("cuda") and torch.cuda.is_available():
            timing["peak_accelerator_memory_bytes"] = float(torch.cuda.max_memory_allocated())
        return metrics, image_rows, threshold_rows, timing

    def _write_prediction(self, root, stem, image, target, probability, index):
        prediction_dir = root / "predictions"
        heatmap_dir = root / "heatmaps"
        overlay_dir = root / "overlays"
        for directory in (prediction_dir, heatmap_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prediction_dir / f"{stem}.png"), probability_to_gray(probability))
        original = tensor_to_rgb(image)
        heatmap = probability_to_heatmap(probability)
        overlay = saliency_overlay(original, probability, self.request.overlay_alpha)
        write_rgb(heatmap_dir / f"{stem}.png", heatmap)
        write_rgb(overlay_dir / f"{stem}.png", overlay)
        if index < 16:
            samples = root / "samples"
            for name in ("original", "ground_truth", "prediction", "overlay", "panels"):
                (samples / name).mkdir(parents=True, exist_ok=True)
            write_rgb(samples / "original" / f"{stem}.png", original)
            cv2.imwrite(str(samples / "ground_truth" / f"{stem}.png"), probability_to_gray(target))
            cv2.imwrite(str(samples / "prediction" / f"{stem}.png"), probability_to_gray(probability))
            write_rgb(samples / "overlay" / f"{stem}.png", overlay)
            write_rgb(
                samples / "panels" / f"{stem}.png",
                compose_saliency_panel(original, target, probability, overlay),
            )

    def _write_artifacts(self, root, metrics, image_rows, threshold_rows, metadata):
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(root / "metrics.json", {"aggregate": metrics})
        write_csv(
            root / "metrics.csv",
            ({"metric": key, "value": value} for key, value in sorted(metrics.items())),
            fieldnames=("metric", "value"),
        )
        write_csv(root / "image_metrics.csv", image_rows)
        write_csv(root / "threshold_metrics.csv", threshold_rows)
        write_csv(
            root / "precision_recall.csv",
            ({key: row[key] for key in ("threshold", "precision", "recall", "f_beta")} for row in threshold_rows),
            fieldnames=("threshold", "precision", "recall", "f_beta"),
        )
        timing = {
            key: value for key, value in metrics.items()
            if "time" in key or "latency" in key or "second" in key or "memory" in key
        }
        write_csv(
            root / "timing.csv",
            ({"metric": key, "value": value} for key, value in sorted(timing.items())),
            fieldnames=("metric", "value"),
        )
        checkpoint = metadata.get("checkpoint_path")
        write_json_atomic(
            root / "run_metadata.json",
            {
                "task": "salient_object_detection",
                "model_name": metadata["model_name"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
                "dataset": self.request.dataset_path,
                "split": self.request.split,
                "input_size": metadata["input_size"],
                "output_channels": 1,
                "activation": "sigmoid",
                "device": self.request.device,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "opencv_version": cv2.__version__,
                "config": self.request.to_config(),
            },
        )
        if self.request.plots:
            self._write_plots(root, image_rows, threshold_rows)

    @staticmethod
    def _write_plots(root, image_rows, threshold_rows):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([row["recall"] for row in threshold_rows], [row["precision"] for row in threshold_rows])
        ax.set(title="Saliency Precision-Recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
        fig.tight_layout()
        fig.savefig(root / "precision_recall.png", dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in ("precision", "recall", "f_beta"):
            ax.plot([row["threshold"] for row in threshold_rows], [row[name] for row in threshold_rows], label=name)
        ax.set(title="Saliency Threshold Metrics", xlabel="Threshold", ylabel="Score", ylim=(0, 1.05))
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "threshold_curves.png", dpi=200)
        plt.close(fig)
        if image_rows:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(
                [[row[name] for row in image_rows] for name in ("mae", "max_f_beta", "mean_f_beta")],
                tick_labels=("MAE", "max F-beta", "mean F-beta"),
            )
            fig.tight_layout()
            fig.savefig(root / "per_image_metric_distributions.png", dpi=200)
            plt.close(fig)

    def _synchronize(self):
        if self.request.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()


BenchmarkFactory = Callable[[BenchmarkSaliencyRequest, WorkflowReporter], BenchmarkSaliencyMapping]


class BenchmarkSaliencyModelGroup:
    def __init__(
        self,
        request: BenchmarkSaliencyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        model_registry=None,
        benchmark_factory: BenchmarkFactory | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.benchmark_factory = benchmark_factory or (
            lambda request, reporter: BenchmarkSaliencyMapping(request, reporter=reporter, **({"model_registry": model_registry} if model_registry is not None else {}))
        )

    def execute(self) -> dict[str, Any]:
        if not self.request.output_path:
            raise MLXUserError("Grouped saliency benchmarking requires --output.")
        root = Path(self.request.output_path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for model_name in grouped_model_names(str(self.request.model), **({"registry": self.model_registry} if self.model_registry is not None else {})):
            checkpoint = self._checkpoint_for(model_name)
            model_request = replace(
                self.request,
                model=model_name,
                model_path=str(checkpoint) if checkpoint else None,
                output_path=str(root / model_name),
            )
            metrics = self.benchmark_factory(model_request, self.reporter).execute()
            rows.append({"model": model_name, **metrics})
        write_csv(root / "comparison.csv", rows)
        result = {"model_group": self.request.model, "models": rows}
        write_json_atomic(root / "comparison.json", result)
        emit(
            self.reporter,
            "success",
            f"Benchmarked {len(rows)} saliency models.",
            payload={"event": "saliency_group_benchmark", **result},
        )
        return result

    def _checkpoint_for(self, model_name: str) -> Path | None:
        if not self.request.model_path:
            return None
        root = Path(self.request.model_path).expanduser()
        candidates = (root / model_name / f"{model_name}.pth", root / f"{model_name}.pth")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise MLXUserError(
            f"No checkpoint for '{model_name}' under grouped --model-path '{root}'. "
            f"Expected '{candidates[0]}' or '{candidates[1]}'."
        )


__all__ = ["BenchmarkSaliencyMapping", "BenchmarkSaliencyModelGroup"]
