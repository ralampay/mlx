"""Post-training slice evaluation for the local YOLOX adapter study."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, replace
import io
import json
import math
from pathlib import Path
import random
import shutil
from statistics import mean, stdev
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_data import load_prepared_adapter_dataset
from mlx.modes.object_detection.adapter_experiment import DEFAULT_DATASET, DEFAULT_OUTPUT, METHODS
from mlx.modes.object_detection.adapter_baselines import recorded_baseline_runs
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    VerifyFoundationCheckpoint,
    require_experiment_device,
)


WEATHER_GROUPS = {
    "low-visibility": ("fog", "haze", "mist"),
    "precipitation": ("rainstorm", "snowstorm"),
    "airborne-particulate": ("sandstorm", "dust-tornado"),
}
COMPARATORS = ("lora", "convpass", "conv-adapter", "bottleneck", "full-finetune", "frozen", "drax")
SLICE_FIELDS = (
    "method", "seed", "slice_family", "slice_name", "image_count", "object_count",
    "low_support", "mAP50", "mAP50_95", "precision", "recall", "AP_small",
    "AP_medium", "AP_large",
)
PAIR_FIELDS = (
    "comparison", "slice_family", "slice_name", "n_seeds", "mean_mAP50_95_difference",
    "standard_deviation", "ci95_low", "ci95_high", "image_quality_difference",
    "image_bootstrap_ci95_low", "image_bootstrap_ci95_high", "bootstrap_samples",
)


@dataclass(frozen=True)
class AdapterSliceRequest:
    """Inputs shared by prediction caching and CPU slice reporting."""

    output: Path
    checkpoint: Path
    dataset: Path
    device: str = "cuda"
    image_size: int = 640
    batch_size: int = 8
    workers: int = 0
    methods: tuple[str, ...] | None = None
    seeds: tuple[int, ...] | None = None
    bootstrap_samples: int = 2_000
    analysis_seed: int = 42
    baseline_study: Path | None = None
    comparison_method: str = "drax"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AdapterSliceRequest":
        explicit = set(config.get("_explicit_options") or ())

        def value(name: str, default: Any) -> Any:
            if "_explicit_options" not in config or name in explicit:
                return config.get(name, default)
            return default

        methods = _parse_strings(config.get("methods"))
        seeds = _parse_ints(config.get("experiment_seeds"))
        return cls(
            output=Path(config.get("output_path") or DEFAULT_OUTPUT).expanduser().resolve(),
            checkpoint=Path(
                config.get("model_path") or "~/Desktop/object-detection-models/foundational-yolox-l.pt"
            ).expanduser().resolve(),
            dataset=Path(value("dataset_path", DEFAULT_DATASET) or DEFAULT_DATASET).expanduser().resolve(),
            device=str(value("device", "cuda") or "cuda"),
            image_size=int(value("height", 640) or 640),
            batch_size=int(value("batch_size", 8) or 8),
            workers=int(value("workers", 0) or 0),
            methods=methods,
            seeds=seeds,
            baseline_study=Path(config["baseline_study"]).expanduser().resolve() if config.get("baseline_study") else None,
            comparison_method=str(config.get("comparison_method") or "drax"),
            bootstrap_samples=int(value("bootstrap_samples", 2_000) or 2_000),
            analysis_seed=int(
                config.get("random_seed") if config.get("random_seed") is not None else 42
            ),
        )

    @property
    def analysis_root(self) -> Path:
        return self.output / "sliced-analysis"

    def validate(self, *, prediction: bool) -> None:
        if not self.output.is_dir():
            raise MLXUserError(f"Adapter study output does not exist: {self.output}")
        if self.bootstrap_samples < 1:
            raise MLXUserError("--bootstrap-samples must be positive")
        if self.comparison_method not in METHODS:
            raise MLXUserError(f"Unknown comparison method: {self.comparison_method}")
        if self.methods and set(self.methods) - set(METHODS):
            raise MLXUserError(f"Slice methods must use {', '.join(METHODS)}")
        if self.seeds and any(seed < 0 for seed in self.seeds):
            raise MLXUserError("Slice seeds must be nonnegative integers")
        if prediction:
            if not self.checkpoint.is_file():
                raise MLXUserError(f"Foundation checkpoint does not exist: {self.checkpoint}")
            if not self.dataset.is_dir():
                raise MLXUserError(f"Prepared adapter dataset does not exist: {self.dataset}")
            if self.image_size < 32 or self.image_size % 32:
                raise MLXUserError("Slice evaluation image size must be a multiple of 32")
            if self.batch_size < 1 or self.workers < 0:
                raise MLXUserError(
                    "Slice evaluation requires positive batch size and nonnegative workers"
                )
            if not self.device.lower().startswith("cuda"):
                raise MLXUserError("Adapter slice prediction generation requires --device cuda")


@dataclass(frozen=True)
class StudyRun:
    method: str
    seed: int
    directory: Path
    metrics: Mapping[str, Any]
    config: Mapping[str, Any]


def _parse_strings(value: Any) -> tuple[str, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    return tuple(dict.fromkeys(part.strip() for part in str(value).split(",") if part.strip()))


def _parse_ints(value: Any) -> tuple[int, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return tuple(dict.fromkeys(int(part.strip()) for part in str(value).split(",")))
    except ValueError as exc:
        raise MLXUserError("--experiment-seeds must be comma-separated integers") from exc


def discover_study_runs(request: AdapterSliceRequest) -> list[StudyRun]:
    """Discover the declared complete study matrix; partial studies are rejected."""
    study_path = request.output / "study.json"
    if not study_path.is_file():
        raise MLXUserError(f"Adapter study metadata is missing: {study_path}")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    baseline = recorded_baseline_runs(request.output, request.baseline_study)
    prior = {(run.method, run.seed): run for run in baseline}
    methods = request.methods or tuple(study.get("available_methods") or ())
    if request.methods is None:
        methods = tuple(dict.fromkeys((*methods, *(run.method for run in baseline))))
    seeds = request.seeds or tuple(int(value) for value in study.get("planned_seeds") or ())
    if not methods or not seeds:
        raise MLXUserError("Study metadata does not declare methods and seeds")
    runs: list[StudyRun] = []
    missing: list[str] = []
    for method in methods:
        for seed in seeds:
            directory = request.output / method / f"seed-{seed}"
            metrics_path = directory / "metrics.json"
            config_path = directory / "config.json"
            if not metrics_path.is_file() or not config_path.is_file():
                if (method, seed) in prior:
                    run = prior[(method, seed)]
                    runs.append(StudyRun(run.method, run.seed, run.directory, run.metrics, run.config))
                    continue
                missing.append(f"{method}/seed-{seed}")
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "completed":
                missing.append(f"{method}/seed-{seed} ({metrics.get('status', 'unknown')})")
                continue
            if metrics.get("method") not in {None, method} or metrics.get("seed") not in {None, seed}:
                raise MLXUserError(f"Run identity differs from its directory: {directory}")
            if config.get("method") not in {None, method} or config.get("seed") not in {None, seed}:
                raise MLXUserError(f"Run configuration identity differs from its directory: {directory}")
            runs.append(StudyRun(method, seed, directory, metrics, config))
    if missing:
        sample = ", ".join(missing[:10])
        suffix = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise MLXUserError(
            f"Slice analysis requires the complete declared study; missing {sample}{suffix}"
        )
    return runs


def weather_category(filename: str) -> str:
    """Return DAWN's canonical weather category from an image filename."""
    stem = Path(filename).stem.lower()
    prefixes = (
        ("sand_storm_g2_", "sandstorm"),
        ("dusttornado-", "dust-tornado"),
        ("rain_storm-", "rainstorm"),
        ("sand_storm-", "sandstorm"),
        ("snow_storm-", "snowstorm"),
        ("foggy-", "fog"),
        ("haze-", "haze"),
        ("mist-", "mist"),
    )
    for prefix, category in prefixes:
        if stem.startswith(prefix):
            return category
    raise MLXUserError(f"Cannot derive a DAWN weather category from {filename!r}")


def weather_group(category: str) -> str:
    for group, categories in WEATHER_GROUPS.items():
        if category in categories:
            return group
    raise MLXUserError(f"No pooled weather group contains {category!r}")


def build_coco_ground_truth(dataset: Path, split: str, classes: Sequence[str]) -> dict[str, Any]:
    """Translate a prepared YOLO split to deterministic COCO ground truth."""
    image_root = dataset / "images" / split
    label_root = dataset / "labels" / split
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in extensions)
    if not image_paths:
        raise MLXUserError(f"No images found in prepared {split} split: {image_root}")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        label_path = label_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise MLXUserError(f"Missing label for {image_path.name}: {label_path}")
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                raise MLXUserError(f"Invalid YOLO label at {label_path}:{line_number}")
            class_value, cx, cy, box_width, box_height = (float(value) for value in fields)
            class_id = int(class_value)
            if class_value != class_id or class_id not in range(len(classes)):
                raise MLXUserError(f"Invalid class ID at {label_path}:{line_number}")
            width_px = box_width * width
            height_px = box_height * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [cx * width - width_px / 2, cy * height - height_px / 2,
                             width_px, height_px],
                    "area": width_px * height_px,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "object"}
            for index, name in enumerate(classes)
        ],
        "info": {"description": f"DAWN prepared {split} split"},
        "licenses": [],
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MLXUserError(f"Cannot read JSON artifact {path}: {exc}") from exc


def _write_compatible_json(path: Path, value: Any) -> None:
    if path.exists() and _load_json(path) != value:
        raise MLXUserError(f"Existing slice artifact is incompatible: {path}")
    write_json_atomic(path, value)


def _prediction_path(root: Path, run: StudyRun, split: str = "test") -> Path:
    return root / "predictions" / run.method / f"seed-{run.seed}" / f"{split}.json"


def _prediction_metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.metadata.json")


def _model_signature(run: StudyRun, foundation_sha: str) -> tuple[str, Path | None]:
    if run.method == "frozen":
        return foundation_sha, None
    if run.method in {"head-only", "full-finetune"}:
        checkpoint = Path(str(run.config.get("selected_checkpoint_path") or ""))
    else:
        checkpoint = run.directory / "adapter" / "checkpoint.pt"
    if not checkpoint.is_file():
        raise MLXUserError(f"Selected checkpoint is missing for {run.method}/seed-{run.seed}: {checkpoint}")
    return sha256_file(checkpoint), checkpoint


class CacheAdapterSlicePredictions:
    """Generate resumable per-image prediction caches without retraining."""

    def __init__(
        self,
        request: AdapterSliceRequest,
        *,
        prediction_writer_factory: Callable[[Mapping[str, Any], torch.device], Callable] | None = None,
    ):
        self.request = request
        self.prediction_writer_factory = prediction_writer_factory

    def execute(self) -> dict[str, Any]:
        request = self.request
        request.validate(prediction=True)
        device = require_experiment_device(request.device)
        model_name = str(_load_json(request.output / "study.json").get("model") or "yolox-l")
        _, foundation = VerifyFoundationCheckpoint(model_name, request.checkpoint).execute()
        dataset = load_prepared_adapter_dataset(request.dataset)
        classes = [foundation["classes"][index] for index in range(foundation["nc"])]
        if dataset["classes"] != classes:
            raise MLXUserError("Prepared dataset class order differs from foundation checkpoint")
        if dataset["selection_sha256"] != _load_json(request.output / "study.json").get(
            "dataset_selection_sha256"
        ):
            raise MLXUserError("Prepared dataset selection differs from the completed study")
        runs = discover_study_runs(request)
        for run in runs:
            if run.metrics.get("checkpoint_sha256") != foundation["sha256"]:
                raise MLXUserError(f"Foundation hash differs for {run.method}/seed-{run.seed}")
            if run.metrics.get("dataset_selection_sha256") != dataset["selection_sha256"]:
                raise MLXUserError(f"Dataset hash differs for {run.method}/seed-{run.seed}")
        root = request.analysis_root
        root.mkdir(parents=True, exist_ok=True)
        ground_truth = {
            split: build_coco_ground_truth(request.dataset, split, classes)
            for split in ("val", "test")
        }
        for split, payload in ground_truth.items():
            _write_compatible_json(root / f"ground_truth_{split}.json", payload)
        for source in {run.directory.parent.parent for run in runs if run.directory.parent.parent != request.output}:
            for split, payload in ground_truth.items():
                if _load_json(source / "sliced-analysis" / f"ground_truth_{split}.json") != payload:
                    raise MLXUserError(f"Baseline {split} ground truth differs: {source}")

        if self.prediction_writer_factory is None:
            from mlx.modes.object_detection.libreyolo.adapter_slice_backend import (
                LibreYOLOAdapterPredictionWriter,
            )

            factory = lambda current, current_device: LibreYOLOAdapterPredictionWriter(
                request, current, current_device
            )
        else:
            factory = self.prediction_writer_factory
        writer = factory(foundation, device)
        signature_cache: dict[tuple[str, str], tuple[Path, Mapping[str, Any]]] = {}
        generated = reused = 0
        frozen_run = next(run for run in runs if run.method == "frozen")
        tasks = [(frozen_run, "val"), *((run, "test") for run in runs)]
        for run, split in tasks:
            signature, checkpoint = _model_signature(run, foundation["sha256"])
            destination = _prediction_path(root, run, split)
            metadata_path = _prediction_metadata_path(destination)
            expected = {
                "schema_version": 1,
                "method": run.method,
                "seed": run.seed,
                "split": split,
                "model_signature": signature,
                "foundation_sha256": foundation["sha256"],
                "dataset_selection_sha256": dataset["selection_sha256"],
                "image_size": request.image_size,
                "confidence": 0.001,
                "iou": 0.6,
                "checkpoint": str(checkpoint) if checkpoint else str(request.checkpoint),
            }
            if run.directory.parent.parent != request.output and not destination.exists():
                source = _prediction_path(run.directory.parent.parent / "sliced-analysis", run, split)
                metadata = _load_json(_prediction_metadata_path(source))
                if any(metadata.get(key) != value for key, value in expected.items()):
                    raise MLXUserError(f"Baseline prediction cache settings differ: {source}")
                if metadata.get("predictions_sha256") != sha256_file(source):
                    raise MLXUserError(f"Baseline prediction cache checksum mismatch: {source}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                write_json_atomic(metadata_path, metadata)
            if destination.is_file() and metadata_path.is_file():
                metadata = _load_json(metadata_path)
                if any(metadata.get(key) != value for key, value in expected.items()):
                    raise MLXUserError(f"Incompatible prediction cache already exists: {destination}")
                if metadata.get("predictions_sha256") != sha256_file(destination):
                    raise MLXUserError(f"Prediction cache checksum mismatch: {destination}")
                signature_cache[(signature, split)] = (destination, metadata)
                reused += 1
                continue
            if destination.exists() or metadata_path.exists():
                raise MLXUserError(
                    f"Incomplete prediction cache exists for {run.method}/seed-{run.seed} {split}; "
                    f"remove both {destination} and {metadata_path} before retrying"
                )
            cached = signature_cache.get((signature, split))
            if cached is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached[0], destination)
                native_metrics = cached[1]["native_metrics"]
                reused += 1
            else:
                native_metrics = writer(run, split, destination)
                generated += 1
            if split == "test":
                recorded = run.metrics
                for native_name, recorded_name in (("map_50", "mAP50"), ("map_50_95", "mAP50_95")):
                    if not math.isclose(
                        float(native_metrics[native_name]), float(recorded[recorded_name]),
                        rel_tol=0.0, abs_tol=5e-4,
                    ):
                        raise MLXUserError(
                            f"Cached {run.method}/seed-{run.seed} {native_name} does not reproduce "
                            f"the study metric: {native_metrics[native_name]} != {recorded[recorded_name]}"
                        )
            metadata = {
                **expected,
                "native_metrics": dict(native_metrics),
                "prediction_count": len(_load_json(destination)),
                "predictions_sha256": sha256_file(destination),
            }
            write_json_atomic(metadata_path, metadata)
            signature_cache[(signature, split)] = (destination, metadata)

        validation_predictions = _load_json(_prediction_path(root, frozen_run, "val"))
        validation_quality = baseline_image_quality(ground_truth["val"], validation_predictions)
        lower, upper = difficulty_tertiles(validation_quality.values())
        manifest = {
            "schema_version": 1,
            "foundation_checkpoint": str(request.checkpoint),
            "comparison_method": request.comparison_method,
            "foundation_sha256": foundation["sha256"],
            "dataset": str(request.dataset),
            "dataset_selection_sha256": dataset["selection_sha256"],
            "methods": list(dict.fromkeys(run.method for run in runs)),
            "seeds": sorted({run.seed for run in runs}),
            "checkpoint_selection": "best-validation",
            "weather_groups": {key: list(value) for key, value in WEATHER_GROUPS.items()},
            "object_size_pixels_squared": {"small_max": 32 ** 2, "medium_max": 96 ** 2},
            "difficulty": {
                "name": "frozen-baseline difficulty",
                "score": "mean over ground truth of max same-class prediction confidence times IoU",
                "cutoffs_from": "validation tertiles",
                "hard_max": lower,
                "medium_max": upper,
            },
            "metric_definitions": {
                "mAP50": "COCO average precision at IoU 0.50",
                "mAP50_95": "COCO average precision over IoU 0.50:0.95",
                "precision": "class-aware greedy precision at confidence 0.25 and IoU 0.50",
                "recall": "class-aware greedy recall at confidence 0.25 and IoU 0.50",
                "image_quality": "mean over ground truth of max same-class confidence times IoU",
            },
            "bootstrap_samples": request.bootstrap_samples,
            "analysis_seed": request.analysis_seed,
        }
        manifest_path = root / "analysis_manifest.json"
        if (manifest_path.exists() and "comparison_method" not in _load_json(manifest_path)
                and request.comparison_method == "drax"):
            manifest.pop("comparison_method")
        if manifest_path.exists() and _load_json(manifest_path) != manifest:
            raise MLXUserError(f"Existing slice manifest differs: {manifest_path}")
        _write_compatible_json(manifest_path, manifest)
        return {
            "runs": len(runs),
            "generated": generated,
            "reused": reused,
            "analysis_root": str(root),
            "manifest": str(manifest_path),
        }


def _xywh_iou(box: Sequence[float], other: Sequence[float]) -> float:
    x1, y1, width1, height1 = box
    x2, y2, width2, height2 = other
    left, top = max(x1, x2), max(y1, y2)
    right, bottom = min(x1 + width1, x2 + width2), min(y1 + height1, y2 + height2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = width1 * height1 + width2 * height2 - intersection
    return intersection / union if union > 0 else 0.0


def baseline_image_quality(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    category_ids: set[int] | None = None,
    area_range: tuple[float, float] | None = None,
) -> dict[int, float]:
    """Continuous frozen-baseline quality used only to define difficulty slices."""
    by_image: dict[int, list[Mapping[str, Any]]] = {}
    for prediction in predictions:
        by_image.setdefault(int(prediction["image_id"]), []).append(prediction)
    quality: dict[int, float] = {}
    annotations: dict[int, list[Mapping[str, Any]]] = {}
    for annotation in ground_truth["annotations"]:
        if category_ids and int(annotation["category_id"]) not in category_ids:
            continue
        if area_range and not area_range[0] <= float(annotation["area"]) < area_range[1]:
            continue
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    for image in ground_truth["images"]:
        image_id = int(image["id"])
        values = []
        for annotation in annotations.get(image_id, []):
            candidates = (
                prediction for prediction in by_image.get(image_id, [])
                if int(prediction["category_id"]) == int(annotation["category_id"])
            )
            values.append(max(
                (float(prediction["score"]) * _xywh_iou(annotation["bbox"], prediction["bbox"])
                 for prediction in candidates),
                default=0.0,
            ))
        if values:
            quality[image_id] = mean(values)
        elif category_ids is None and area_range is None:
            quality[image_id] = 0.0
    return quality


def difficulty_tertiles(values: Iterable[float]) -> tuple[float, float]:
    materialized = np.asarray(tuple(float(value) for value in values), dtype=float)
    if materialized.size < 3 or not np.isfinite(materialized).all():
        raise MLXUserError("Difficulty tertiles require at least three finite validation scores")
    lower, upper = np.quantile(materialized, (1 / 3, 2 / 3), method="linear")
    return float(lower), float(upper)


def difficulty_label(value: float, lower: float, upper: float) -> str:
    return "hard" if value <= lower else "medium" if value <= upper else "easy"


def _fixed_precision_recall(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    image_ids: set[int],
    *,
    category_ids: set[int] | None = None,
    area_range: tuple[float, float] | None = None,
    confidence: float = 0.25,
    match_iou: float = 0.5,
) -> tuple[float, float]:
    annotations: dict[int, list[Mapping[str, Any]]] = {}
    for annotation in ground_truth["annotations"]:
        image_id = int(annotation["image_id"])
        area = float(annotation["area"])
        if image_id not in image_ids or (category_ids and int(annotation["category_id"]) not in category_ids):
            continue
        if area_range and not area_range[0] <= area < area_range[1]:
            continue
        annotations.setdefault(image_id, []).append(annotation)
    detections: dict[int, list[Mapping[str, Any]]] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if image_id in image_ids and float(prediction["score"]) >= confidence:
            if not category_ids or int(prediction["category_id"]) in category_ids:
                if area_range:
                    prediction_area = float(prediction["bbox"][2]) * float(prediction["bbox"][3])
                    if not area_range[0] <= prediction_area < area_range[1]:
                        continue
                detections.setdefault(image_id, []).append(prediction)
    true_positive = false_positive = false_negative = 0
    for image_id in image_ids:
        gt = annotations.get(image_id, [])
        matched: set[int] = set()
        for prediction in sorted(detections.get(image_id, []), key=lambda row: float(row["score"]), reverse=True):
            candidates = [
                index for index, annotation in enumerate(gt)
                if index not in matched
                and int(annotation["category_id"]) == int(prediction["category_id"])
            ]
            if not candidates:
                false_positive += 1
                continue
            overlaps = [_xywh_iou(gt[index]["bbox"], prediction["bbox"]) for index in candidates]
            best = max(range(len(overlaps)), key=overlaps.__getitem__)
            if overlaps[best] >= match_iou:
                matched.add(candidates[best])
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(gt) - len(matched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall


def _coco_metrics(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    image_ids: set[int],
    *,
    category_ids: set[int] | None = None,
    area_label: str = "all",
) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not image_ids:
        return {key: 0.0 for key in ("mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large")}
    with redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dict(ground_truth)
        coco.createIndex()
        selected = [row for row in predictions if int(row["image_id"]) in image_ids]
        if not selected:
            return {key: 0.0 for key in ("mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large")}
        result = coco.loadRes(selected)
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        if category_ids:
            evaluator.params.catIds = sorted(category_ids)
        evaluator.evaluate()
        evaluator.accumulate()
    area_index = {"all": 0, "small": 1, "medium": 2, "large": 3}[area_label]
    precision = evaluator.eval["precision"][:, :, :, area_index, -1]
    valid = precision[precision > -1]
    ap = float(valid.mean()) if valid.size else 0.0
    ap50_values = precision[0]
    ap50_valid = ap50_values[ap50_values > -1]
    ap50 = float(ap50_valid.mean()) if ap50_valid.size else 0.0
    size_values = {}
    for label, index in (("small", 1), ("medium", 2), ("large", 3)):
        values = evaluator.eval["precision"][:, :, :, index, -1]
        values = values[values > -1]
        size_values[label] = float(values.mean()) if values.size else 0.0
    return {
        "mAP50": ap50,
        "mAP50_95": ap,
        "AP_small": size_values["small"],
        "AP_medium": size_values["medium"],
        "AP_large": size_values["large"],
    }


def _slice_row(
    run: StudyRun,
    family: str,
    name: str,
    image_ids: set[int],
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    category_ids: set[int] | None = None,
    area_label: str = "all",
) -> dict[str, Any]:
    area_ranges = {"small": (0.0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, math.inf)}
    area_range = area_ranges.get(area_label)
    objects = [
        annotation for annotation in ground_truth["annotations"]
        if int(annotation["image_id"]) in image_ids
        and (not category_ids or int(annotation["category_id"]) in category_ids)
        and (not area_range or area_range[0] <= float(annotation["area"]) < area_range[1])
    ]
    metrics = _coco_metrics(
        ground_truth, predictions, image_ids, category_ids=category_ids, area_label=area_label
    )
    precision, recall = _fixed_precision_recall(
        ground_truth, predictions, image_ids, category_ids=category_ids, area_range=area_range
    )
    return {
        "method": run.method,
        "seed": run.seed,
        "slice_family": family,
        "slice_name": name,
        "image_count": len(image_ids),
        "object_count": len(objects),
        "low_support": len(image_ids) < 10 or len(objects) < 10,
        **metrics,
        "precision": precision,
        "recall": recall,
    }


def _t_interval(values: Sequence[float]) -> tuple[float, float]:
    average = mean(values)
    if len(values) < 2:
        return float("nan"), float("nan")
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    margin = critical * stdev(values) / math.sqrt(len(values))
    return average - margin, average + margin


def _bootstrap_mean_interval(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    draws = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return _quantile(draws, 0.025), _quantile(draws, 0.975)


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1 - fraction) + values[upper] * fraction)


class GenerateAdapterSliceReport:
    """Create weather, size, class, difficulty, and paired reports from caches."""

    def __init__(self, request: AdapterSliceRequest):
        self.request = request

    def execute(self) -> dict[str, Any]:
        request = self.request
        request.validate(prediction=False)
        root = request.analysis_root
        manifest_path = root / "analysis_manifest.json"
        if not manifest_path.is_file():
            raise MLXUserError(
                f"Slice predictions are not prepared: {manifest_path}. Run adapter-slice-predict first."
            )
        manifest = _load_json(manifest_path)
        if request.comparison_method != manifest.get("comparison_method", "drax"):
            raise MLXUserError("--comparison-method differs from the fixed analysis manifest")
        if request.bootstrap_samples != int(manifest.get("bootstrap_samples", request.bootstrap_samples)):
            raise MLXUserError("--bootstrap-samples differs from the fixed analysis manifest")
        if request.analysis_seed != int(manifest.get("analysis_seed", request.analysis_seed)):
            raise MLXUserError("--seed differs from the fixed analysis manifest")
        discovery_request = replace(
            request,
            methods=request.methods or tuple(manifest.get("methods") or ()),
            seeds=request.seeds or tuple(int(value) for value in manifest.get("seeds") or ()),
        )
        runs = discover_study_runs(discovery_request)
        test_gt = _load_json(root / "ground_truth_test.json")
        lower = float(manifest["difficulty"]["hard_max"])
        upper = float(manifest["difficulty"]["medium_max"])
        images = {int(row["id"]): row for row in test_gt["images"]}
        all_ids = set(images)
        frozen = next(run for run in runs if run.method == "frozen")
        frozen_predictions = _load_json(_prediction_path(root, frozen, "test"))
        test_quality = baseline_image_quality(test_gt, frozen_predictions)
        assignments = []
        annotations_by_image: dict[int, list[Mapping[str, Any]]] = {}
        for annotation in test_gt["annotations"]:
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image_id, image in images.items():
            weather = weather_category(str(image["file_name"]))
            annotations = annotations_by_image.get(image_id, [])
            assignments.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "weather": weather,
                    "weather_group": weather_group(weather),
                    "difficulty": difficulty_label(test_quality[image_id], lower, upper),
                    "frozen_quality": test_quality[image_id],
                    "objects": len(annotations),
                    "small_objects": sum(float(row["area"]) < 32 ** 2 for row in annotations),
                    "medium_objects": sum(32 ** 2 <= float(row["area"]) < 96 ** 2 for row in annotations),
                    "large_objects": sum(float(row["area"]) >= 96 ** 2 for row in annotations),
                }
            )
        write_csv(root / "slice_assignments.csv", assignments)
        ids_by_weather = {
            name: {row["image_id"] for row in assignments if row["weather"] == name}
            for name in sorted({row["weather"] for row in assignments})
        }
        ids_by_group = {
            name: {row["image_id"] for row in assignments if row["weather_group"] == name}
            for name in WEATHER_GROUPS
        }
        ids_by_difficulty = {
            name: {row["image_id"] for row in assignments if row["difficulty"] == name}
            for name in ("hard", "medium", "easy")
        }
        categories = {int(row["id"]): str(row["name"]) for row in test_gt["categories"]}
        slice_specs: list[tuple[str, str, set[int], set[int] | None, str]] = [
            ("overall", "all", all_ids, None, "all"),
            *(("weather", name, ids, None, "all") for name, ids in ids_by_weather.items()),
            *(("weather_group", name, ids, None, "all") for name, ids in ids_by_group.items()),
            *(("difficulty", name, ids, None, "all") for name, ids in ids_by_difficulty.items()),
            *(("object_size", name, all_ids, None, name) for name in ("small", "medium", "large")),
            *(("class", name, all_ids, {category_id}, "all") for category_id, name in categories.items()),
        ]
        for weather, image_ids in ids_by_weather.items():
            for category_id, category in categories.items():
                object_count = sum(
                    int(annotation["image_id"]) in image_ids
                    and int(annotation["category_id"]) == category_id
                    for annotation in test_gt["annotations"]
                )
                if len(image_ids) >= 5 and object_count >= 10:
                    slice_specs.append(
                        ("class_weather", f"{category}@{weather}", image_ids, {category_id}, "all")
                    )

        rows: list[dict[str, Any]] = []
        prediction_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for run in runs:
            path = _prediction_path(root, run, "test")
            metadata = _prediction_metadata_path(path)
            if not path.is_file() or not metadata.is_file():
                raise MLXUserError(f"Missing prediction cache for {run.method}/seed-{run.seed}")
            if _load_json(metadata).get("predictions_sha256") != sha256_file(path):
                raise MLXUserError(f"Prediction cache checksum mismatch: {path}")
            predictions = _load_json(path)
            prediction_cache[(run.method, run.seed)] = predictions
            for family, name, image_ids, category_ids, area_label in slice_specs:
                rows.append(
                    _slice_row(
                        run, family, name, image_ids, test_gt, predictions,
                        category_ids=category_ids, area_label=area_label,
                    )
                )
        write_csv(root / "slice_results.csv", rows, fieldnames=SLICE_FIELDS)
        write_csv(
            root / "weather-results.csv",
            [row for row in rows if row["slice_family"] in {"weather", "weather_group"}],
            fieldnames=SLICE_FIELDS,
        )
        write_csv(
            root / "object-size-results.csv",
            [row for row in rows if row["slice_family"] == "object_size"],
            fieldnames=SLICE_FIELDS,
        )
        write_csv(
            root / "class-weather-results.csv",
            [row for row in rows if row["slice_family"] == "class_weather"],
            fieldnames=SLICE_FIELDS,
        )
        paired = self._paired_results(
            rows, slice_specs, test_gt, prediction_cache,
            request.bootstrap_samples, request.analysis_seed,
            candidate=request.comparison_method,
        )
        write_csv(root / "paired_differences.csv", paired, fieldnames=PAIR_FIELDS)
        results = {
            "manifest": manifest,
            "runs": len(runs),
            "slice_rows": len(rows),
            "slice_results": rows,
            "slice_definitions": [
                {"family": family, "name": name, "image_count": len(image_ids)}
                for family, name, image_ids, _categories, _area in slice_specs
            ],
            "paired_differences": paired,
        }
        write_json_atomic(root / "results.json", results)
        self._write_summary(root / "summary.md", rows, paired, candidate=request.comparison_method)
        return {
            "runs": len(runs),
            "slice_rows": len(rows),
            "analysis_root": str(root),
            "results": str(root / "results.json"),
            "summary": str(root / "summary.md"),
        }

    @staticmethod
    def _paired_results(rows, slice_specs, ground_truth, prediction_cache, samples, seed, *, candidate="drax"):
        by_key = {
            (row["method"], int(row["seed"]), row["slice_family"], row["slice_name"]): row
            for row in rows
        }
        seeds = sorted({int(row["seed"]) for row in rows if row["method"] == candidate})
        if not seeds:
            raise MLXUserError(f"No completed candidate runs for {candidate}")
        quality_cache: dict[tuple[str, int, tuple[int, ...], str], dict[int, float]] = {}

        def qualities(method, run_seed, categories, area_label):
            category_key = tuple(sorted(categories or ()))
            key = (method, run_seed, category_key, area_label)
            if key not in quality_cache:
                area_ranges = {
                    "small": (0.0, 32 ** 2),
                    "medium": (32 ** 2, 96 ** 2),
                    "large": (96 ** 2, math.inf),
                }
                quality_cache[key] = baseline_image_quality(
                    ground_truth,
                    prediction_cache[(method, run_seed)],
                    category_ids=categories,
                    area_range=area_ranges.get(area_label),
                )
            return quality_cache[key]

        paired = []
        for comparator in COMPARATORS:
            if comparator == candidate:
                continue
            if not all((comparator, value) in prediction_cache for value in seeds):
                continue
            for index, (family, name, image_ids, categories, area_label) in enumerate(slice_specs):
                differences = [
                    float(by_key[(candidate, value, family, name)]["mAP50_95"])
                    - float(by_key[(comparator, value, family, name)]["mAP50_95"])
                    for value in seeds
                ]
                ci_low, ci_high = _t_interval(differences)
                drax_quality = {
                    value: qualities(candidate, value, categories, area_label) for value in seeds
                }
                comparator_quality = {
                    value: qualities(comparator, value, categories, area_label) for value in seeds
                }
                eligible_ids = [
                    image_id for image_id in sorted(image_ids)
                    if all(
                        image_id in drax_quality[value]
                        and image_id in comparator_quality[value]
                        for value in seeds
                    )
                ]
                image_differences = [
                    mean(drax_quality[value][image_id] for value in seeds)
                    - mean(comparator_quality[value][image_id] for value in seeds)
                    for image_id in eligible_ids
                ]
                bootstrap_low, bootstrap_high = _bootstrap_mean_interval(
                    image_differences, samples, seed + index
                )
                paired.append(
                    {
                        "comparison": f"{candidate}-minus-{comparator}",
                        "slice_family": family,
                        "slice_name": name,
                        "n_seeds": len(seeds),
                        "mean_mAP50_95_difference": mean(differences),
                        "standard_deviation": stdev(differences) if len(differences) > 1 else 0.0,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "image_quality_difference": mean(image_differences) if image_differences else None,
                        "image_bootstrap_ci95_low": bootstrap_low,
                        "image_bootstrap_ci95_high": bootstrap_high,
                        "bootstrap_samples": samples,
                    }
                )
        return paired

    @staticmethod
    def _write_summary(path: Path, rows, paired, *, candidate="drax") -> None:
        overall = [row for row in rows if row["slice_family"] == "overall"]
        grouped: dict[str, list[float]] = {}
        for row in overall:
            grouped.setdefault(str(row["method"]), []).append(float(row["mAP50_95"]))
        lines = [
            "# Targeted adapter performance", "",
            "All values use validation-selected checkpoints. Weather slices with fewer than ten images "
            "are descriptive and marked low support.", "",
            "| Method | Seeds | Overall mAP50-95 |", "|---|---:|---:|",
        ]
        for method, values in sorted(grouped.items()):
            lines.append(f"| {method} | {len(values)} | {mean(values):.4f} |")
        lines.extend([
            "", f"## {candidate} targeted slices", "",
            "| Slice family | Slice | Images | Objects | Mean mAP50-95 | Support |",
            "|---|---|---:|---:|---:|---|",
        ])
        drax_slices: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            if row["method"] == candidate and row["slice_family"] in {
                "weather", "weather_group", "difficulty", "object_size"
            }:
                drax_slices.setdefault((row["slice_family"], row["slice_name"]), []).append(row)
        for (family, name), values in sorted(drax_slices.items()):
            support = "low" if any(bool(row["low_support"]) for row in values) else "adequate"
            lines.append(
                f"| {family} | {name} | {values[0]['image_count']} | {values[0]['object_count']} | "
                f"{mean(float(row['mAP50_95']) for row in values):.4f} | {support} |"
            )
        lines.extend(["", f"## {candidate} paired comparisons", ""])
        for row in paired:
            if row["slice_family"] != "overall":
                continue
            lines.append(
                f"- {row['comparison']}: {row['mean_mAP50_95_difference']:+.4f} mAP50-95 "
                f"(95% seed-level CI {row['ci95_low']:+.4f} to {row['ci95_high']:+.4f})."
            )
        lines.extend([
            "", "The difficulty slices are frozen-baseline difficulty, not a causal measure of domain-shift severity.",
            "Non-significance is not evidence of equivalence.", "",
        ])
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(path)


__all__ = [
    "AdapterSliceRequest",
    "CacheAdapterSlicePredictions",
    "GenerateAdapterSliceReport",
    "StudyRun",
    "baseline_image_quality",
    "build_coco_ground_truth",
    "difficulty_label",
    "difficulty_tertiles",
    "discover_study_runs",
    "weather_category",
    "weather_group",
]
