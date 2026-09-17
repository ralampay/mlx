from __future__ import annotations

import csv
import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.data import resolve_optional_segmentation_test_split
from mlx.modes.segmentation.evaluation import BenchmarkSegmentation
from mlx.modes.segmentation.models import MODEL_GROUP_NAMES, grouped_model_names
from mlx.modes.segmentation.requests import (
    BenchmarkSegmentationRequest,
    TrainSegmentationRequest,
)
from mlx.modes.segmentation.train import TrainSegmentationModel
from mlx.modes.segmentation.utils import resolve_train_output_paths


RANKING_METRIC = "mean_foreground_dice"
LEADERBOARD_METRICS = (
    "mean_foreground_dice",
    "mean_foreground_iou",
    "macro_dice",
    "mean_iou",
    "pixel_accuracy",
    "cross_entropy_loss",
    "generalized_dice",
    "cohen_kappa",
    "images_per_second_forward",
)


@dataclass(frozen=True)
class SegmentationModelEvaluation:
    model_name: str
    output_dir: Path
    checkpoint_path: Path
    benchmark_dir: Path
    best_validation_loss: float
    metrics: Mapping[str, float]

    def leaderboard_row(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "model": self.model_name,
            "best_validation_loss": self.best_validation_loss,
            **{name: self.metrics.get(name) for name in LEADERBOARD_METRICS},
            "checkpoint": self.checkpoint_path,
            "output_dir": self.output_dir,
            "benchmark_dir": self.benchmark_dir,
        }


@dataclass(frozen=True)
class SegmentationAllModelsResult:
    status: str
    output_dir: Path
    ranking_metric: str
    results: tuple[SegmentationModelEvaluation, ...]


TrainerFactory = Callable[
    [TrainSegmentationRequest, WorkflowReporter],
    TrainSegmentationModel,
]
BenchmarkFactory = Callable[
    [BenchmarkSegmentationRequest, WorkflowReporter],
    BenchmarkSegmentation,
]


class TrainAllSegmentationModels:
    def __init__(
        self,
        request: TrainSegmentationRequest,
        *,
        reporter: WorkflowReporter | None = None,
        trainer_factory: TrainerFactory | None = None,
        benchmark_factory: BenchmarkFactory | None = None,
        model_names: list[str] | tuple[str, ...] | None = None,
        allow_dataset_source_manifest: bool = False,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.trainer_factory = trainer_factory or _default_trainer_factory
        self.benchmark_factory = benchmark_factory or _default_benchmark_factory
        self.model_names = tuple(
            sorted(
                model_names
                if model_names is not None
                else grouped_model_names(str(request.model))
            )
        )
        self.allow_dataset_source_manifest = allow_dataset_source_manifest

    def execute(self) -> SegmentationAllModelsResult:
        output_root = validate_all_models_request(
            self.request,
            allow_dataset_source_manifest=self.allow_dataset_source_manifest,
        )
        test_split = resolve_optional_segmentation_test_split(
            self.request.dataset_path
        )
        if test_split is None:
            raise MLXUserError(
                f"Segmentation --model {self.request.model} requires a test/images and test/masks "
                "partition for post-training benchmarking."
            )
        if not self.model_names:
            raise MLXUserError(
                f"No segmentation models are registered for --model {self.request.model}."
            )

        output_root.mkdir(parents=True, exist_ok=True)
        completed: list[SegmentationModelEvaluation] = []
        emit(
            self.reporter,
            "info",
            f"Training and benchmarking {len(self.model_names)} segmentation models.",
            current=0,
            total=len(self.model_names),
            payload={"event": "segmentation_all_models_start"},
        )
        for index, model_name in enumerate(self.model_names, start=1):
            emit(
                self.reporter,
                "progress",
                f"Starting segmentation model {model_name} ({index}/{len(self.model_names)}).",
                current=index - 1,
                total=len(self.model_names),
                payload={
                    "event": "segmentation_all_models_model_start",
                    "model": model_name,
                },
            )
            try:
                result = self._train_and_benchmark(
                    model_name,
                    output_root,
                )
                completed.append(result)
                self._write_results(
                    output_root,
                    completed,
                    status="running",
                    current_model=model_name,
                )
            except Exception as exc:
                self._write_results(
                    output_root,
                    completed,
                    status="failed",
                    current_model=model_name,
                    error=str(exc),
                )
                raise MLXUserError(
                    f"All-model segmentation run failed for '{model_name}': {exc}"
                ) from exc
            finally:
                _release_accelerator_memory()

        ranked = _rank_results(completed)
        self._write_results(output_root, ranked, status="completed")
        emit(
            self.reporter,
            "success",
            f"Completed all {len(ranked)} segmentation models.",
            current=len(ranked),
            total=len(ranked),
            payload={
                "event": "segmentation_all_models_complete",
                "rows": [
                    result.leaderboard_row(rank)
                    for rank, result in enumerate(ranked, start=1)
                ],
                "output_dir": str(output_root),
            },
        )
        return SegmentationAllModelsResult(
            status="completed",
            output_dir=output_root,
            ranking_metric=RANKING_METRIC,
            results=tuple(ranked),
        )

    def _train_and_benchmark(
        self,
        model_name: str,
        output_root: Path,
    ) -> SegmentationModelEvaluation:
        model_output = output_root / model_name
        model_request = replace(
            self.request,
            model=model_name,
            model_path=None,
            output_path=str(model_output),
        )
        self.trainer_factory(model_request, self.reporter).execute()
        paths = resolve_train_output_paths(
            model_request.to_config(),
            model_name=model_name,
        )
        checkpoint_path = paths["checkpoint_path"]
        if not checkpoint_path.is_file():
            raise MLXUserError(
                f"Training model '{model_name}' did not produce its best-validation-loss "
                f"checkpoint at '{checkpoint_path}'."
            )
        best_validation_loss = _read_best_validation_loss(
            paths["training_csv_path"]
        )

        benchmark_dir = model_output / "benchmark"
        benchmark_request = BenchmarkSegmentationRequest.from_config(
            {
                **model_request.to_config(),
                "action": "benchmark",
                "model_path": str(checkpoint_path),
                "output_path": str(benchmark_dir),
                "split": "test",
            }
        )
        metrics = self.benchmark_factory(
            benchmark_request,
            self.reporter,
        ).execute()
        return SegmentationModelEvaluation(
            model_name=model_name,
            output_dir=model_output,
            checkpoint_path=checkpoint_path,
            benchmark_dir=benchmark_dir,
            best_validation_loss=best_validation_loss,
            metrics=dict(metrics),
        )

    def _write_results(
        self,
        output_root: Path,
        results: list[SegmentationModelEvaluation],
        *,
        status: str,
        current_model: str | None = None,
        error: str | None = None,
    ) -> None:
        ranked = _rank_results(results)
        write_json_atomic(
            output_root / "all-models.json",
            {
                "status": status,
                "ranking_metric": RANKING_METRIC,
                "models": list(self.model_names),
                "completed_models": len(results),
                "current_model": current_model,
                "error": error,
                "results": [asdict(result) for result in ranked],
            },
        )
        write_csv(
            output_root / "leaderboard.csv",
            [
                result.leaderboard_row(rank)
                for rank, result in enumerate(ranked, start=1)
            ],
            fieldnames=(
                "rank",
                "model",
                "best_validation_loss",
                *LEADERBOARD_METRICS,
                "checkpoint",
                "output_dir",
                "benchmark_dir",
            ),
        )


def validate_all_models_request(
    request: TrainSegmentationRequest,
    *,
    allow_dataset_source_manifest: bool = False,
) -> Path:
    if request.model not in MODEL_GROUP_NAMES:
        available = ", ".join(f"--model {name}" for name in sorted(MODEL_GROUP_NAMES))
        raise MLXUserError(f"TrainAllSegmentationModels requires {available}.")
    if request.pretrained:
        raise MLXUserError(
            f"Segmentation --model {request.model} requires scratch initialization; "
            "omit --pretrained."
        )
    if request.model_path:
        raise MLXUserError(
            f"Segmentation --model {request.model} cannot use one --model-path "
            "across multiple models."
        )
    if not request.output_path:
        raise MLXUserError(
            f"Segmentation --model {request.model} requires --output pointing to "
            "a new artifact directory."
        )
    output_root = Path(request.output_path).expanduser()
    if output_root.suffix.lower() in {".pt", ".pth"}:
        raise MLXUserError(
            f"Segmentation --model {request.model} requires a directory output, "
            "not a checkpoint file."
        )
    if output_root.exists() and not output_root.is_dir():
        raise MLXUserError(
            f"Segmentation grouped-model output must be a directory: {output_root}"
        )
    if output_root.is_dir():
        entries = list(output_root.iterdir())
        if allow_dataset_source_manifest:
            entries = [path for path in entries if path.name != "dataset_source.json"]
        if entries:
            raise MLXUserError(
                f"Segmentation grouped-model output directory must be empty: {output_root}"
            )
    return output_root


def _rank_results(
    results: list[SegmentationModelEvaluation],
) -> list[SegmentationModelEvaluation]:
    def sort_key(result: SegmentationModelEvaluation) -> tuple[int, float, str]:
        value = float(result.metrics.get(RANKING_METRIC, float("nan")))
        if not np.isfinite(value):
            return (1, 0.0, result.model_name)
        return (0, -value, result.model_name)

    return sorted(results, key=sort_key)


def _read_best_validation_loss(path: Path) -> float:
    if not path.is_file():
        raise MLXUserError(
            f"Segmentation training did not produce its history at '{path}'."
        )
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            values = [
                float(row["val_loss"])
                for row in csv.DictReader(input_file)
                if row.get("val_loss") not in (None, "")
            ]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise MLXUserError(
            f"Unable to read validation loss history from '{path}': {exc}"
        ) from exc
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        raise MLXUserError(
            f"Segmentation training history '{path}' contains no finite validation loss."
        )
    return min(finite)


def _default_trainer_factory(
    request: TrainSegmentationRequest,
    reporter: WorkflowReporter,
) -> TrainSegmentationModel:
    return TrainSegmentationModel(request, reporter=reporter)


def _default_benchmark_factory(
    request: BenchmarkSegmentationRequest,
    reporter: WorkflowReporter,
) -> BenchmarkSegmentation:
    return BenchmarkSegmentation(request, reporter=reporter)


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


__all__ = [
    "LEADERBOARD_METRICS",
    "RANKING_METRIC",
    "SegmentationAllModelsResult",
    "SegmentationModelEvaluation",
    "TrainAllSegmentationModels",
    "validate_all_models_request",
]
