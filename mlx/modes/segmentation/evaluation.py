from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from rich.table import Table
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_success
from mlx.modes.segmentation.data import (
    SegmentationEvaluationDataset,
    resolve_segmentation_evaluation_split,
)
from mlx.modes.segmentation.metrics import (
    SegmentationResearchMetrics,
    aggregate_boundary_metrics,
    aggregate_confusion_metrics,
    class_metrics_from_confusion,
    confusion_matrix_from_arrays,
    per_image_metrics,
    probability_metrics,
    threshold_sweep,
)
from mlx.modes.segmentation.presentation import blend_overlay, colorize_mask
from mlx.modes.segmentation.research import (
    write_class_metrics_plot,
    write_confusion_matrix_artifacts,
    write_csv,
    write_curve_artifacts,
    write_image_metric_distribution,
    write_json,
    write_metrics_csv,
    write_threshold_artifacts,
)
from mlx.modes.segmentation.utils import load_checkpoint_bundle, resolve_class_names


class BenchmarkSegmentation:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = str(config.get("device", "cpu"))
        self.split = str(config.get("split", "test"))
        self.boundary_tolerance = int(config.get("boundary_tolerance", 2))
        self.calibration_bins = int(config.get("calibration_bins", 15))
        self.threshold_steps = int(config.get("threshold_steps", 101))

    def execute(self) -> dict[str, float]:
        self._validate_config()
        model, metadata = load_checkpoint_bundle(self.config)
        model = model.to(self.device)
        model.eval()
        class_names = resolve_class_names(
            self.config,
            int(metadata["num_classes"]),
            checkpoint_names=metadata.get("class_names"),
        )
        evaluation_path = resolve_segmentation_evaluation_split(
            self.config["dataset_path"],
            split=self.split,
        )
        dataset = SegmentationEvaluationDataset(
            evaluation_path,
            input_size=tuple(metadata["input_size"]),
            num_classes=int(metadata["num_classes"]),
            colored=bool(metadata["colored"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.config.get("batch_size", 1))),
            shuffle=False,
            num_workers=2,
        )
        result, timing = self._evaluate(
            model,
            metadata,
            dataset,
            loader,
            class_names,
        )
        result.metrics.update(timing)
        self._render(result.metrics, result.class_rows)
        output_dir = self._output_dir()
        if output_dir is not None:
            self._write_artifacts(
                output_dir,
                result,
                metadata,
                evaluation_path,
                class_names,
            )
        return result.metrics

    def _validate_config(self) -> None:
        threshold = float(self.config.get("mask_threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise MLXUserError("--mask-threshold must be between 0 and 1.")
        if self.boundary_tolerance < 0:
            raise MLXUserError("--boundary-tolerance must be zero or greater.")
        if self.calibration_bins < 2:
            raise MLXUserError("--calibration-bins must be at least 2.")
        if self.threshold_steps < 2:
            raise MLXUserError("--threshold-steps must be at least 2.")

    def _evaluate(
        self,
        model,
        metadata: dict[str, Any],
        dataset: SegmentationEvaluationDataset,
        loader: DataLoader,
        class_names: list[str],
    ) -> tuple[SegmentationResearchMetrics, dict[str, float]]:
        num_classes = len(class_names)
        configured_threshold = float(
            self.config.get("mask_threshold", metadata.get("mask_threshold", 0.5))
        )
        criterion = nn.CrossEntropyLoss(reduction="sum")
        target_batches: list[np.ndarray] = []
        prediction_batches: list[np.ndarray] = []
        probability_batches: list[np.ndarray] = []
        image_rows: list[dict[str, Any]] = []
        forward_times: list[float] = []
        total_loss = 0.0
        sample_offset = 0
        wall_start = time.perf_counter()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Benchmarking segmentation"):
                images = images.to(self.device)
                masks_device = masks.to(self.device)
                self._synchronize()
                forward_start = time.perf_counter()
                logits = model(images)
                self._synchronize()
                elapsed = time.perf_counter() - forward_start
                forward_times.append(elapsed)
                total_loss += float(criterion(logits, masks_device).item())

                probabilities = torch.softmax(logits, dim=1)
                if num_classes == 2:
                    predictions = (probabilities[:, 1] >= configured_threshold).long()
                else:
                    predictions = logits.argmax(dim=1)
                targets_np = masks.numpy().astype(np.int64, copy=False)
                predictions_np = predictions.cpu().numpy().astype(np.int64, copy=False)
                probabilities_np = probabilities.permute(0, 2, 3, 1).cpu().numpy()
                target_batches.append(targets_np)
                prediction_batches.append(predictions_np)
                probability_batches.append(probabilities_np)

                per_sample_time = elapsed / max(1, len(images))
                for batch_index in range(len(images)):
                    image_path, mask_path = dataset.samples[sample_offset + batch_index]
                    row: dict[str, Any] = {
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "width": int(targets_np[batch_index].shape[1]),
                        "height": int(targets_np[batch_index].shape[0]),
                        "inference_ms": per_sample_time * 1000.0,
                    }
                    sample_probabilities = probabilities_np[batch_index]
                    sample_target = targets_np[batch_index]
                    selected_probabilities = sample_probabilities[
                        np.arange(sample_target.shape[0])[:, None],
                        np.arange(sample_target.shape[1])[None, :],
                        sample_target,
                    ]
                    row.update(
                        {
                            "cross_entropy_loss": float(
                                -np.log(np.clip(selected_probabilities, 1e-12, 1.0)).mean()
                            ),
                            "mean_confidence": float(
                                sample_probabilities.max(axis=-1).mean()
                            ),
                            "mean_predictive_entropy": float(
                                np.mean(
                                    -np.sum(
                                        sample_probabilities
                                        * np.log(
                                            np.clip(sample_probabilities, 1e-12, 1.0)
                                        ),
                                        axis=-1,
                                    )
                                )
                            ),
                        }
                    )
                    row.update(
                        per_image_metrics(
                            targets_np[batch_index],
                            predictions_np[batch_index],
                            class_names=class_names,
                            boundary_tolerance=self.boundary_tolerance,
                        )
                    )
                    image_rows.append(row)
                    if self._output_dir() is not None and bool(
                        self.config.get("save_images", True)
                    ):
                        self._write_prediction_images(
                            self._output_dir(),
                            image_path,
                            targets_np[batch_index],
                            predictions_np[batch_index],
                            metadata,
                        )
                sample_offset += len(images)

        wall_seconds = time.perf_counter() - wall_start
        targets = np.concatenate(target_batches, axis=0)
        predictions = np.concatenate(prediction_batches, axis=0)
        probabilities = np.concatenate(probability_batches, axis=0)
        matrix = confusion_matrix_from_arrays(targets, predictions, num_classes)
        class_rows = class_metrics_from_confusion(matrix, class_names)
        metrics = aggregate_confusion_metrics(matrix, class_rows)
        metrics["cross_entropy_loss"] = total_loss / max(1, targets.size)
        metrics["multiclass_mcc"] = float(
            matthews_corrcoef(targets.reshape(-1), predictions.reshape(-1))
        )
        probability_summary, curves = probability_metrics(
            targets,
            probabilities,
            class_rows,
            calibration_bins=self.calibration_bins,
        )
        metrics.update(probability_summary)
        metrics.update(aggregate_boundary_metrics(image_rows, class_names))
        threshold_rows: list[dict[str, float]] = []
        if num_classes == 2:
            threshold_rows, threshold_summary = threshold_sweep(
                targets,
                probabilities[..., 1],
                threshold_steps=self.threshold_steps,
                configured_threshold=configured_threshold,
            )
            metrics.update(threshold_summary)

        timing = self._timing_metrics(
            wall_seconds=wall_seconds,
            forward_times=forward_times,
            per_image_times=[
                float(row["inference_ms"]) / 1000.0 for row in image_rows
            ],
            image_count=len(dataset),
            pixel_count=int(targets.size),
        )
        return (
            SegmentationResearchMetrics(
                metrics=metrics,
                class_rows=class_rows,
                image_rows=image_rows,
                confusion_matrix=matrix,
                curves=curves,
                threshold_rows=threshold_rows,
            ),
            timing,
        )

    def _timing_metrics(
        self,
        *,
        wall_seconds: float,
        forward_times: list[float],
        per_image_times: list[float],
        image_count: int,
        pixel_count: int,
    ) -> dict[str, float]:
        total_forward = float(sum(forward_times))
        per_batch_ms = np.asarray(forward_times, dtype=np.float64) * 1000.0
        per_image_ms = np.asarray(per_image_times, dtype=np.float64) * 1000.0
        result = {
            "evaluated_images": float(image_count),
            "evaluated_pixels": float(pixel_count),
            "wall_time_seconds": wall_seconds,
            "forward_time_seconds": total_forward,
            "images_per_second_wall": image_count / wall_seconds if wall_seconds else float("nan"),
            "images_per_second_forward": image_count / total_forward
            if total_forward
            else float("nan"),
            "mean_batch_latency_ms": float(per_batch_ms.mean()),
            "median_batch_latency_ms": float(np.median(per_batch_ms)),
            "p95_batch_latency_ms": float(np.percentile(per_batch_ms, 95)),
            "p99_batch_latency_ms": float(np.percentile(per_batch_ms, 99)),
            "mean_image_latency_ms": float(per_image_ms.mean()),
            "median_image_latency_ms": float(np.median(per_image_ms)),
            "p95_image_latency_ms": float(np.percentile(per_image_ms, 95)),
            "p99_image_latency_ms": float(np.percentile(per_image_ms, 99)),
        }
        if self.device.startswith("cuda") and torch.cuda.is_available():
            result["peak_accelerator_memory_bytes"] = float(torch.cuda.max_memory_allocated())
        return result

    def _write_prediction_images(
        self,
        output_dir: Path,
        image_path: Path,
        target: np.ndarray,
        prediction: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        masks_dir = output_dir / "predictions" / "masks"
        overlays_dir = output_dir / "predictions" / "overlays"
        errors_dir = output_dir / "predictions" / "errors"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        errors_dir.mkdir(parents=True, exist_ok=True)
        mask_dtype = np.uint8 if int(metadata["num_classes"]) <= 256 else np.uint16
        cv2.imwrite(str(masks_dir / f"{image_path.stem}.png"), prediction.astype(mask_dtype))

        original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise MLXUserError(f"Unable to read benchmark image for overlay: {image_path}")
        original_rgb = cv2.cvtColor(
            cv2.resize(original_bgr, tuple(metadata["input_size"])),
            cv2.COLOR_BGR2RGB,
        )
        color_mask = colorize_mask(prediction.astype(np.uint8), metadata["palette"])
        overlay = blend_overlay(
            original_rgb,
            color_mask,
            float(self.config.get("overlay_alpha", 0.45)),
        )
        cv2.imwrite(
            str(overlays_dir / f"{image_path.stem}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        error = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        if int(metadata["num_classes"]) == 2:
            error[(target == 0) & (prediction == 1)] = (255, 0, 0)
            error[(target == 1) & (prediction == 0)] = (0, 0, 255)
        else:
            error[target != prediction] = (255, 0, 255)
        cv2.imwrite(
            str(errors_dir / f"{image_path.stem}.png"),
            cv2.cvtColor(error, cv2.COLOR_RGB2BGR),
        )

    def _write_artifacts(
        self,
        output_dir: Path,
        result: SegmentationResearchMetrics,
        metadata: dict[str, Any],
        evaluation_path: Path,
        class_names: list[str],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_metrics_csv(output_dir / "metrics.csv", result.metrics)
        write_json(
            output_dir / "metrics.json",
            {
                "aggregate": result.metrics,
                "classes": result.class_rows,
            },
        )
        write_csv(output_dir / "class_metrics.csv", result.class_rows)
        write_csv(output_dir / "image_metrics.csv", result.image_rows)
        timing_keys = [
            key
            for key in result.metrics
            if "time" in key
            or "latency" in key
            or "second" in key
            or "memory" in key
            or key.startswith("evaluated_")
        ]
        write_metrics_csv(
            output_dir / "timing.csv",
            {key: result.metrics[key] for key in timing_keys},
        )
        write_csv(
            output_dir / "worst_cases.csv",
            sorted(
                result.image_rows,
                key=lambda row: np.nan_to_num(
                    float(row.get("mean_foreground_dice", float("nan"))),
                    nan=-1.0,
                ),
            ),
        )
        write_confusion_matrix_artifacts(
            output_dir,
            result.confusion_matrix,
            class_names,
        )
        write_curve_artifacts(output_dir, result.curves, class_names)
        write_class_metrics_plot(output_dir, result.class_rows)
        write_image_metric_distribution(output_dir, result.image_rows)
        write_threshold_artifacts(output_dir, result.threshold_rows)
        checkpoint_path = Path(metadata["checkpoint_path"])
        write_json(
            output_dir / "run_metadata.json",
            {
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "dataset": evaluation_path,
                "requested_split": self.split,
                "class_names": class_names,
                "input_size": metadata["input_size"],
                "num_classes": metadata["num_classes"],
                "device": self.device,
                "device_name": self._device_name(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "opencv_version": cv2.__version__,
                "config": self.config,
            },
        )
        print_success(f"Segmentation research artifacts written to {output_dir}")

    def _render(self, metrics: dict[str, float], class_rows: list[dict[str, Any]]) -> None:
        table = Table(title="Segmentation Benchmark Results", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        for key in (
            "cross_entropy_loss",
            "pixel_accuracy",
            "macro_dice",
            "mean_foreground_dice",
            "mean_iou",
            "mean_foreground_iou",
            "frequency_weighted_iou",
            "cohen_kappa",
            "multiclass_mcc",
            "macro_roc_auc",
            "macro_average_precision",
            "expected_calibration_error",
            "images_per_second_forward",
        ):
            table.add_row(key, f"{metrics[key]:.6f}")
        console.print(table)
        class_table = Table(title="Per-Class Segmentation Metrics")
        for heading in ("Class", "Precision", "Recall", "Specificity", "Dice", "IoU", "ROC AUC"):
            class_table.add_column(heading, justify="right" if heading != "Class" else "left")
        for row in class_rows:
            class_table.add_row(
                str(row["class_name"]),
                _format_metric(row.get("precision")),
                _format_metric(row.get("recall")),
                _format_metric(row.get("specificity")),
                _format_metric(row.get("dice")),
                _format_metric(row.get("iou")),
                _format_metric(row.get("roc_auc")),
            )
        console.print(class_table)

    def _synchronize(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _device_name(self) -> str:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
        return self.device

    def _output_dir(self) -> Path | None:
        output_path = self.config.get("output_path")
        return Path(output_path).expanduser() if output_path else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    return f"{numeric:.4f}" if np.isfinite(numeric) else "nan"
