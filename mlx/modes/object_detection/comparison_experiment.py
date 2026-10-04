"""Paired low-data comparison of YOLOX-M and CSP-Drax feature fusion."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import yaml

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.core.paired_statistics import AnalyzePairedDifferences
from mlx.modes.object_detection.evaluation import normalize_detection_metrics
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    FineTuneObjectDetectionRequest,
)


CONTROL = "yolox-m"
CANDIDATE = "yolox-drax-csp-fusion-m"
MODELS = (CONTROL, CANDIDATE)


@dataclass(frozen=True)
class CSPDraxComparisonRequest:
    dataset: Path
    output: Path
    source_checkpoint: Path
    seeds: tuple[int, ...] = (17, 29, 43, 59, 71, 89, 101, 127)
    initialization_seed: int = 20261003
    epochs: int = 100
    batch_size: int = 8
    image_size: int = 640
    workers: int = 4
    learning_rate: float = 0.001
    warmup_epochs: float = 5.0
    no_aug_epochs: int = 15
    confidence: float = 0.001
    iou: float = 0.6
    max_detections: int = 300
    equivalence_margin: float = 0.02
    bootstrap_draws: int = 100_000
    device: str = "0"
    amp: bool = True

    def normalized(self) -> "CSPDraxComparisonRequest":
        return CSPDraxComparisonRequest(
            **{
                **asdict(self),
                "dataset": self.dataset.expanduser().resolve(),
                "output": self.output.expanduser().resolve(),
                "source_checkpoint": self.source_checkpoint.expanduser().resolve(),
            }
        )

    def validate(self, *, require_source: bool = True) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise MLXUserError(
                "Comparison seeds must be unique and contain at least two values."
            )
        if min(self.epochs, self.batch_size, self.image_size, self.bootstrap_draws) < 1:
            raise MLXUserError(
                "Epochs, batch size, image size, and bootstrap draws must be positive."
            )
        if self.image_size % 32:
            raise MLXUserError("Comparison image size must be divisible by 32.")
        if self.learning_rate <= 0 or self.equivalence_margin <= 0:
            raise MLXUserError("Learning rate and equivalence margin must be positive.")
        if not (self.dataset / "data.yaml").is_file():
            raise MLXUserError(f"Prepared dataset YAML not found: {self.dataset / 'data.yaml'}")
        if require_source and not self.source_checkpoint.is_file():
            raise MLXUserError(
                f"YOLOX-M source checkpoint not found: {self.source_checkpoint}"
            )


class PrepareCSPDraxComparison:
    """Create paired initial checkpoints from one pretrained YOLOX-M body."""

    def __init__(self, request: CSPDraxComparisonRequest):
        self.request = request.normalized()

    def execute(self) -> dict[str, Any]:
        request = self.request
        request.validate(require_source=True)
        class_count = self._dataset_class_count(request.dataset / "data.yaml")
        dataset_record = self._dataset_record(request.dataset)
        try:
            from libreyolo import LibreYOLOX, LibreYOLOXDraxCSPFusionM
            from libreyolo.models.yolox_drax_csp_fusion import (
                InitializeYOLOXSharedTransfer,
            )
        except ImportError as exc:
            raise MLXUserError(
                "The CSP-Drax comparison requires a LibreYOLO checkout exposing "
                "LibreYOLOXDraxCSPFusionM."
            ) from exc

        control = LibreYOLOX(None, size="m", nb_classes=class_count, device="cpu")
        candidate = LibreYOLOXDraxCSPFusionM(
            None, size="m", nb_classes=class_count, device="cpu"
        )
        report = InitializeYOLOXSharedTransfer(
            request.source_checkpoint,
            control,
            candidate,
            reset_seed=request.initialization_seed,
        ).execute()
        initial = request.output / "initialization"
        control_path = Path(control.save(str(initial / "yolox-m.pt"))).resolve()
        candidate_path = Path(
            candidate.save(str(initial / "yolox-drax-csp-fusion-m.pt"))
        ).resolve()
        control_parameters = sum(p.numel() for p in control.model.parameters())
        candidate_parameters = sum(p.numel() for p in candidate.model.parameters())
        if candidate_parameters >= control_parameters:
            raise MLXUserError(
                "CSP-Drax fusion exceeds the YOLOX-M parameter budget: "
                f"{candidate_parameters:,} >= {control_parameters:,}."
            )
        manifest = {
            "schema_version": 1,
            "created_at": _now(),
            "source_checkpoint": str(request.source_checkpoint),
            "source_sha256": sha256_file(request.source_checkpoint),
            "source_url": (
                "https://github.com/Megvii-BaseDetection/YOLOX/releases/"
                "download/0.1.1rc0/yolox_m.pth"
            ),
            "dataset": dataset_record,
            "classes": class_count,
            "transfer": report.to_dict() if hasattr(report, "to_dict") else asdict(report),
            "models": {
                CONTROL: self._checkpoint_record(control_path, control_parameters),
                CANDIDATE: self._checkpoint_record(candidate_path, candidate_parameters),
            },
            "protocol": asdict(request),
        }
        write_json_atomic(initial / "manifest.json", manifest)
        return manifest

    @staticmethod
    def _dataset_class_count(path: Path) -> int:
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise MLXUserError(f"Cannot read comparison dataset YAML {path}: {exc}") from exc
        names = content.get("names")
        if isinstance(names, dict):
            count = len(names)
        elif isinstance(names, list):
            count = len(names)
        else:
            count = 0
        if count < 1:
            raise MLXUserError(f"Dataset YAML has no classes: {path}")
        return count

    @staticmethod
    def _checkpoint_record(path: Path, parameters: int) -> dict[str, Any]:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "parameters": parameters,
            "bytes": path.stat().st_size,
        }

    @staticmethod
    def _dataset_record(root: Path) -> dict[str, Any]:
        expected_counts = {"train": 1000, "val": 500, "test": 4952}
        manifests = {}
        for split, expected_count in expected_counts.items():
            path = root / f"{split}_manifest.json"
            if not path.is_file():
                raise MLXUserError(f"Dataset split manifest not found: {path}")
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MLXUserError(f"Cannot read dataset split manifest {path}: {exc}") from exc
            if not isinstance(rows, list) or len(rows) != expected_count:
                actual = len(rows) if isinstance(rows, list) else "invalid"
                raise MLXUserError(
                    f"Expected {expected_count} {split} images, found {actual} in {path}."
                )
            manifests[split] = {
                "images": len(rows),
                "sha256": sha256_file(path),
            }
        return {"path": str(root), "splits": manifests}


class RunCSPDraxComparison:
    """Run balanced paired seeds and evaluate validation-selected checkpoints."""

    def __init__(
        self,
        request: CSPDraxComparisonRequest,
        *,
        train: Callable[[FineTuneObjectDetectionRequest], Any] | None = None,
        benchmark: Callable[[BenchmarkObjectDetectionRequest], Any] | None = None,
    ) -> None:
        self.request = request.normalized()
        self.train = train or self._train
        self.benchmark = benchmark or self._benchmark

    def execute(self, *, smoke: bool = False) -> list[dict[str, Any]]:
        request = self.request
        request.validate(require_source=False)
        manifest = self._load_initialization()
        records = []
        schedule = self._balanced_schedule(request.seeds[:1] if smoke else request.seeds)
        for seed, model in schedule:
            run_dir = request.output / ("smoke" if smoke else "runs") / model / f"seed-{seed}"
            completed = run_dir / "complete.json"
            if completed.is_file():
                existing = json.loads(completed.read_text(encoding="utf-8"))
                self._validate_existing(existing, manifest, model, seed, smoke)
                records.append(existing)
                continue
            initial = Path(manifest["models"][model]["path"])
            started = time.perf_counter()
            training = self.train(self._training_request(model, seed, initial, run_dir, smoke))
            checkpoint = self._checkpoint_path(training)
            training_seconds = time.perf_counter() - started
            benchmark_started = time.perf_counter()
            evaluation = self.benchmark(
                self._benchmark_request(model, checkpoint, run_dir, smoke)
            )
            native_metrics = getattr(evaluation, "metrics", evaluation)
            record = {
                "schema_version": 1,
                "completed_at": _now(),
                "model": model,
                "seed": seed,
                "smoke": smoke,
                "initial_checkpoint": str(initial),
                "initial_sha256": sha256_file(initial),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameters": manifest["models"][model]["parameters"],
                "training_seconds": training_seconds,
                "benchmark_seconds": time.perf_counter() - benchmark_started,
                "metrics": normalize_detection_metrics(native_metrics),
                "effective_batch_size": 2 if smoke else request.batch_size,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": None,
                "epochs": 1 if smoke else request.epochs,
                "image_size": request.image_size,
                "learning_rate": request.learning_rate,
            }
            write_json_atomic(completed, record)
            records.append(record)
        return records

    def _validate_existing(
        self,
        record: dict[str, Any],
        manifest: dict[str, Any],
        model: str,
        seed: int,
        smoke: bool,
    ) -> None:
        initialization = manifest["models"][model]
        expected = {
            "model": model,
            "seed": seed,
            "smoke": smoke,
            "initial_sha256": initialization["sha256"],
            "effective_batch_size": 2 if smoke else self.request.batch_size,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": None,
            "epochs": 1 if smoke else self.request.epochs,
            "image_size": self.request.image_size,
            "learning_rate": self.request.learning_rate,
        }
        mismatched = [
            name for name, value in expected.items() if record.get(name) != value
        ]
        if mismatched:
            raise MLXUserError(
                f"Existing comparison run is incompatible at model={model}, "
                f"seed={seed}: {', '.join(mismatched)}."
            )

    @staticmethod
    def _balanced_schedule(seeds: tuple[int, ...]) -> list[tuple[int, str]]:
        return [
            (seed, model)
            for index, seed in enumerate(seeds)
            for model in (MODELS if index % 2 == 0 else tuple(reversed(MODELS)))
        ]

    def _load_initialization(self) -> dict[str, Any]:
        path = self.request.output / "initialization/manifest.json"
        if not path.is_file():
            raise MLXUserError(
                f"Initialization manifest not found: {path}. Run prepare first."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for model in MODELS:
            record = manifest.get("models", {}).get(model, {})
            checkpoint = Path(record.get("path", ""))
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != record.get("sha256")
            ):
                raise MLXUserError(
                    f"Initialization checkpoint is missing or changed for {model}."
                )
        return manifest

    def _training_request(
        self, model: str, seed: int, initial: Path, run_dir: Path, smoke: bool
    ) -> FineTuneObjectDetectionRequest:
        batch = 2 if smoke else self.request.batch_size
        return FineTuneObjectDetectionRequest(
            provider="libreyolo",
            model=model,
            model_path=str(initial),
            dataset_path=str(self.request.dataset),
            output_path=str(run_dir / "training"),
            run_name="formal",
            device=self.request.device,
            height=self.request.image_size,
            width=self.request.image_size,
            epochs=1 if smoke else self.request.epochs,
            batch_size=batch,
            optimizer="sgd",
            nbs=batch,
            warmup_epochs=0.0 if smoke else self.request.warmup_epochs,
            no_aug_epochs=0 if smoke else self.request.no_aug_epochs,
            patience=1 if smoke else self.request.epochs,
            lr0=self.request.learning_rate,
            amp=self.request.amp,
            random_seed=seed,
            workers=0 if smoke else self.request.workers,
            eval_interval=1,
            plots=not smoke,
        )

    def _benchmark_request(
        self, model: str, checkpoint: Path, run_dir: Path, smoke: bool
    ) -> BenchmarkObjectDetectionRequest:
        return BenchmarkObjectDetectionRequest(
            provider="libreyolo",
            model=model,
            model_path=str(checkpoint),
            dataset_path=str(self.request.dataset),
            output_path=str(run_dir / "benchmark"),
            split="test",
            device=self.request.device,
            height=self.request.image_size,
            width=self.request.image_size,
            batch_size=2 if smoke else self.request.batch_size,
            confidence=self.request.confidence,
            iou=self.request.iou,
            max_detections=self.request.max_detections,
            workers=0 if smoke else self.request.workers,
            save_predictions=True,
            plots=not smoke,
        )

    @staticmethod
    def _checkpoint_path(result: Any) -> Path:
        value = (
            result.get("checkpoint_path") or result.get("model_path")
            if isinstance(result, dict)
            else getattr(result, "checkpoint_path", None)
        )
        path = Path(value or "").expanduser().resolve()
        if not path.is_file():
            raise MLXUserError("Comparison training did not return a usable checkpoint.")
        return path

    @staticmethod
    def _train(request: FineTuneObjectDetectionRequest):
        from mlx.modes.object_detection.commands import FineTuneObjectDetectionModel

        return FineTuneObjectDetectionModel(request).execute()

    @staticmethod
    def _benchmark(request: BenchmarkObjectDetectionRequest):
        from mlx.modes.object_detection.commands import BenchmarkObjectDetectionModel

        return BenchmarkObjectDetectionModel(request).execute()


class AnalyzeCSPDraxComparison:
    """Aggregate completed pairs and write the statistical study report."""

    def __init__(self, request: CSPDraxComparisonRequest):
        self.request = request.normalized()

    def execute(self) -> dict[str, Any]:
        request = self.request
        records = []
        missing = []
        for seed in request.seeds:
            for model in MODELS:
                path = request.output / "runs" / model / f"seed-{seed}/complete.json"
                if path.is_file():
                    records.append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    missing.append({"model": model, "seed": seed})
        complete = [
            seed
            for seed in request.seeds
            if all(
                any(
                    row["seed"] == seed and row["model"] == model
                    for row in records
                )
                for model in MODELS
            )
        ]
        analysis: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": _now(),
            "primary_metric": "map_50_95",
            "difference": "candidate minus control",
            "complete_pairs": len(complete),
            "planned_pairs": len(request.seeds),
            "missing": missing,
            "decision": "incomplete",
        }
        if len(complete) == len(request.seeds):
            indexed = {(row["seed"], row["model"]): row for row in records}
            differences = [
                indexed[seed, CANDIDATE]["metrics"]["map_50_95"]
                - indexed[seed, CONTROL]["metrics"]["map_50_95"]
                for seed in request.seeds
            ]
            result = AnalyzePairedDifferences(
                differences,
                equivalence_margin=request.equivalence_margin,
                bootstrap_draws=request.bootstrap_draws,
            ).execute()
            analysis.update(result.to_dict())
            analysis["paired_differences"] = differences

        results_dir = request.output / "results"
        rows = [
            {
                "model": row["model"],
                "seed": row["seed"],
                **row["metrics"],
                "parameters": row["parameters"],
                "training_seconds": row["training_seconds"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in records
        ]
        write_csv(results_dir / "results.csv", rows)
        write_json_atomic(results_dir / "statistics.json", analysis)
        report = self._render_report(records, analysis)
        request.output.mkdir(parents=True, exist_ok=True)
        (request.output / "REPORT.md").write_text(report, encoding="utf-8")
        return analysis

    def _render_report(self, records: list[dict], analysis: dict) -> str:
        labels = {
            "favor_candidate": "The fusion candidate is significantly better.",
            "favor_control": "YOLOX-M is significantly better.",
            "practically_equivalent": "The models are practically equivalent within ±0.02 AP.",
            "inconclusive": "Neither superiority nor equivalence was established.",
            "incomplete": (
                "The formal experiment is incomplete; no inferential conclusion is "
                "reported."
            ),
        }
        lines = [
            "# CSP-Drax fusion vs YOLOX-M on low-data VOC07",
            "",
            f"Generated: {_now()}",
            "",
            "## Status",
            "",
            "Completed paired replicates: "
            f"**{analysis['complete_pairs']} / {analysis['planned_pairs']}**.",
            "",
            f"**Decision:** {labels[analysis['decision']]}",
        ]
        if analysis["decision"] != "incomplete":
            low, high = analysis["bootstrap_95_ci"]
            lines.extend([
                "",
                "Mean paired AP50-95 difference (candidate − control): "
                f"**{analysis['mean_difference']:+.4f}**; exact two-sided p = "
                f"**{analysis['exact_sign_flip_p_two_sided']:.4g}**; "
                f"paired bootstrap 95% CI **[{low:+.4f}, {high:+.4f}]**.",
                "",
                f"TOST ±{analysis['equivalence_margin']:.2f}: lower p = "
                f"{analysis['tost_lower_p']:.4g}, upper p = "
                f"{analysis['tost_upper_p']:.4g}; equivalence = "
                f"{analysis['equivalent']}.",
            ])
        lines.extend([
            "",
            "## Aggregate results",
            "",
            "| Model | Runs | AP50-95 | AP50 | Parameters | Mean training hours |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for model in MODELS:
            values = [row for row in records if row["model"] == model]
            if values:
                lines.append(
                    f"| {model} | {len(values)} | "
                    f"{mean(row['metrics']['map_50_95'] for row in values):.4f} | "
                    f"{mean(row['metrics']['map_50'] for row in values):.4f} | "
                    f"{values[0]['parameters']:,} | "
                    f"{mean(row['training_seconds'] for row in values) / 3600:.2f} |"
                )
        lines.extend([
            "",
            "## Method",
            "",
            (
                "Both models receive the same pretrained YOLOX-M backbone/PAN "
                "tensors. Both detection heads are reset from the same recorded "
                "seed; candidate-only fusion and Drax parameters retain their "
                "identity-oriented initialization. The protocol uses 1,000 "
                "training images, 500 validation images, all 4,952 VOC07 test "
                "images, 100 epochs, physical and effective batch size 8, no "
                "gradient accumulation, no gradient clipping, and eight paired "
                "seeds."
            ),
            "",
            (
                "The primary test is an exact paired sign-flip test on AP50-95. "
                "A paired bootstrap interval describes uncertainty. TOST uses a "
                "prespecified ±0.02 AP equivalence margin; non-significance is "
                "not interpreted as equivalence."
            ),
            "",
            "## Invalidated pilot",
            "",
            (
                "The earlier scratch-trained pilot is excluded from inference "
                "because only CSP-Drax used norm-1 gradient clipping and `nbs=64` "
                "turned physical batch 8 into eight-step accumulation. It tested "
                "a different optimization regime and a legacy graph without "
                "cross-scale attention fusion."
            ),
            "",
            "## Limitations",
            "",
            (
                "Inference is conditional on one deterministic low-data VOC07 "
                "subset, eight paired seeds, COCO-style evaluation of VOC boxes, "
                "and one software/hardware environment. VOC difficult flags are "
                "treated as ordinary objects."
            ),
            "",
        ])
        return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
