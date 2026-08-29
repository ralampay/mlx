from __future__ import annotations

import csv
import json
import math
import platform
import time
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from mlx.core.exceptions import MLXUserError
from mlx.core.artifacts import sha256_file as _sha256
from mlx.modes.object_detection.models import ObjectDetectionBenchmarkResult
from mlx.modes.object_detection.requests import BenchmarkObjectDetectionRequest


STANDARD_METRIC_KEYS = ("precision", "recall", "f1", "map_50", "map_50_95")

_METRIC_ALIASES = {
    "precision": ("metrics/precision", "metrics/precision(B)", "precision"),
    "recall": ("metrics/recall", "metrics/recall(B)", "recall"),
    "map_50": ("metrics/mAP50", "metrics/mAP50(B)", "mAP50", "map_50"),
    "map_50_95": (
        "metrics/mAP50-95",
        "metrics/mAP50-95(B)",
        "metrics/mAP50_95",
        "mAP50_95",
        "map_50_95",
    ),
}


def validate_benchmark_request(request: BenchmarkObjectDetectionRequest) -> None:
    if not request.model_path:
        raise MLXUserError("Object-detection benchmarking requires --model-path.")
    model_path = Path(request.model_path).expanduser()
    if not model_path.is_file():
        raise MLXUserError(f"Object-detection model not found: {model_path}")
    if request.split not in {"train", "val", "test"}:
        raise MLXUserError("Object-detection benchmark split must be train, val, or test.")
    if request.batch_size < 1:
        raise MLXUserError("--batch-size must be at least 1 for benchmarking.")
    if request.workers < 0:
        raise MLXUserError("--workers must be zero or greater.")
    if request.max_detections < 1:
        raise MLXUserError("--max-detections must be at least 1.")
    if not 0.0 <= request.confidence <= 1.0:
        raise MLXUserError("--confidence must be between 0 and 1.")
    if not 0.0 < request.iou <= 1.0:
        raise MLXUserError("--iou must be greater than 0 and at most 1.")


def resolve_benchmark_output_dir(request: BenchmarkObjectDetectionRequest) -> Path:
    if request.output_path:
        return Path(request.output_path).expanduser().resolve()
    checkpoint = Path(str(request.model_path)).expanduser().resolve()
    return checkpoint.parent / "benchmarks" / request.split


def scalar_metrics(values: Mapping[str, Any] | Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        result_dict = getattr(values, "results_dict", None)
        if callable(result_dict):
            result_dict = result_dict()
        values = result_dict if isinstance(result_dict, Mapping) else {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, Real) and math.isfinite(float(value))
    }


def normalize_detection_metrics(values: Mapping[str, Any] | Any) -> dict[str, float]:
    native = scalar_metrics(values)
    metrics: dict[str, float] = {}
    for standard_name, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias in native:
                metrics[standard_name] = native[alias]
                break
    if "precision" in metrics and "recall" in metrics:
        denominator = metrics["precision"] + metrics["recall"]
        metrics["f1"] = (
            2.0 * metrics["precision"] * metrics["recall"] / denominator
            if denominator
            else 0.0
        )
    missing = [key for key in STANDARD_METRIC_KEYS if key not in metrics]
    if missing:
        raise MLXUserError(
            "The provider did not return the required benchmark metric(s): "
            f"{', '.join(missing)}."
        )
    return {key: metrics[key] for key in STANDARD_METRIC_KEYS}


def write_benchmark_artifacts(
    *,
    request: BenchmarkObjectDetectionRequest,
    provider: str,
    dataset: str,
    evaluation_backend: str,
    raw_metrics: Mapping[str, Any] | Any,
    started_at: datetime,
    started_clock: float,
    provider_version: str | None = None,
) -> ObjectDetectionBenchmarkResult:
    output_dir = resolve_benchmark_output_dir(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = normalize_detection_metrics(raw_metrics)
    native_metrics = scalar_metrics(raw_metrics)
    completed_at = datetime.now(timezone.utc)

    _write_json(output_dir / "metrics.json", metrics)
    _write_metrics_csv(output_dir / "metrics.csv", metrics)
    _write_json(output_dir / "native_metrics.json", native_metrics)
    model_path = Path(str(request.model_path)).expanduser().resolve()
    _write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "provider": provider,
            "provider_version": provider_version,
            "evaluation_backend": evaluation_backend,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "dataset": dataset,
            "split": request.split,
            "image_size": [request.height, request.width],
            "batch_size": request.batch_size,
            "confidence": request.confidence,
            "iou": request.iou,
            "max_detections": request.max_detections,
            "metric_definitions": {
                "precision": "provider aggregate detection precision",
                "recall": "provider aggregate detection recall",
                "f1": "harmonic mean of the normalized aggregate precision and recall",
                "map_50": "mean average precision at IoU 0.50",
                "map_50_95": "mean average precision over IoU 0.50:0.95",
            },
            "device": request.device,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": time.perf_counter() - started_clock,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    return ObjectDetectionBenchmarkResult(
        provider=provider,
        model_path=str(model_path),
        dataset=dataset,
        split=request.split,
        metrics=metrics,
        output_dir=output_dir,
        evaluation_backend=evaluation_backend,
        native_metrics=native_metrics,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _write_metrics_csv(path: Path, metrics: Mapping[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in STANDARD_METRIC_KEYS:
            writer.writerow({"metric": key, "value": metrics[key]})
