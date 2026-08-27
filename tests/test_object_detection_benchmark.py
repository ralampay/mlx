from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection import runner
from mlx.modes.object_detection.aws.entrypoint import RunSageMakerObjectDetectionTraining
from mlx.modes.object_detection.commands import (
    BenchmarkObjectDetectionModel,
    TrainObjectDetectionModel,
)
from mlx.modes.object_detection.models import (
    ObjectDetectionBenchmarkResult,
    ObjectDetectionTrainingResult,
)
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    TrainObjectDetectionRequest,
)


STANDARD_METRICS = {
    "precision": 0.8,
    "recall": 0.5,
    "f1": 2 * 0.8 * 0.5 / 1.3,
    "map_50": 0.7,
    "map_50_95": 0.4,
}


class FakeBenchmarkProvider:
    name = "fake"

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self.training_requests = []
        self.benchmark_requests = []

    def train(self, request, reporter):
        self.training_requests.append(request)
        assert self.checkpoint is not None
        return {"checkpoint_path": str(self.checkpoint)}

    def benchmark(self, request, reporter):
        self.benchmark_requests.append(request)
        output_dir = Path(request.output_path)
        return ObjectDetectionBenchmarkResult(
            provider=self.name,
            model_path=str(request.model_path),
            dataset=request.dataset_path,
            split=request.split,
            metrics=STANDARD_METRICS,
            output_dir=output_dir,
            evaluation_backend="fake-evaluator",
            native_metrics=STANDARD_METRICS,
        )


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {0: thing}\n",
        encoding="utf-8",
    )
    return dataset


def test_neutral_benchmark_command_delegates_to_provider(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    provider = FakeBenchmarkProvider()
    request = BenchmarkObjectDetectionRequest(
        provider="fake",
        model_path=str(checkpoint),
        dataset_path=str(_dataset(tmp_path)),
        output_path=str(tmp_path / "benchmark"),
    )

    result = BenchmarkObjectDetectionModel(request, provider=provider).execute()

    assert result.metrics == STANDARD_METRICS
    assert provider.benchmark_requests == [request]


def test_training_can_compose_post_training_benchmark(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "trial" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    dataset = _dataset(tmp_path)
    provider = FakeBenchmarkProvider(checkpoint)

    result = TrainObjectDetectionModel(
        TrainObjectDetectionRequest(
            provider="fake",
            model="model",
            dataset_path=str(dataset),
            batch_size=4,
            validate_after_training=True,
            validation_split="val",
            validation_confidence=0.002,
            validation_iou=0.55,
            validation_max_detections=100,
        ),
        provider=provider,
    ).execute()

    benchmark_request = provider.benchmark_requests[0]
    assert result.checkpoint_path == str(checkpoint.resolve())
    assert benchmark_request.model_path == str(checkpoint.resolve())
    assert benchmark_request.split == "val"
    assert benchmark_request.confidence == pytest.approx(0.002)
    assert benchmark_request.iou == pytest.approx(0.55)
    assert benchmark_request.max_detections == 100
    assert benchmark_request.output_path == str(
        checkpoint.parent.parent / "benchmark" / "val"
    )


def test_invalid_post_training_benchmark_is_rejected_before_training(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    provider = FakeBenchmarkProvider(checkpoint)

    with pytest.raises(MLXUserError, match="validation split"):
        TrainObjectDetectionModel(
            TrainObjectDetectionRequest(
                provider="fake",
                validate_after_training=True,
                validation_split="holdout",
            ),
            provider=provider,
        ).execute()

    assert provider.benchmark_requests == []
    assert provider.training_requests == []


def test_libreyolo_benchmark_writes_normalized_research_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    from mlx.modes.object_detection.libreyolo.evaluation import (
        BenchmarkLibreYOLOObjectDetection,
    )

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = _dataset(tmp_path)
    calls = []

    class FakeModel:
        last_eval_backend = "pycocotools 2.0"

        def val(self, **kwargs):
            calls.append(kwargs)
            return {
                "metrics/precision": 0.8,
                "metrics/recall": 0.5,
                "metrics/mAP50": 0.7,
                "metrics/mAP50-95": 0.4,
            }

    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *args, **kwargs: FakeModel()),
    )
    output_dir = tmp_path / "libreyolo-benchmark"
    result = BenchmarkLibreYOLOObjectDetection(
        BenchmarkObjectDetectionRequest(
            provider="libreyolo",
            model_path=str(checkpoint),
            dataset_path=str(dataset),
            output_path=str(output_dir),
            confidence=0.001,
            split="test",
        )
    ).execute()

    assert result.metrics == pytest.approx(STANDARD_METRICS)
    assert result.evaluation_backend == "pycocotools 2.0"
    assert calls[0]["split"] == "test"
    assert calls[0]["save_dir"] == str(output_dir.resolve())
    assert json.loads((output_dir / "metrics.json").read_text()) == pytest.approx(
        STANDARD_METRICS
    )
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["provider"] == "libreyolo"
    assert metadata["model_sha256"]
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "native_metrics.json").is_file()


def test_ultralytics_benchmark_uses_same_normalized_artifact_contract(
    monkeypatch, tmp_path: Path
) -> None:
    from mlx.modes.object_detection.ultralytics import evaluation

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = _dataset(tmp_path)
    calls = []

    class FakeModel:
        def val(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                results_dict={
                    "metrics/precision(B)": 0.8,
                    "metrics/recall(B)": 0.5,
                    "metrics/mAP50(B)": 0.7,
                    "metrics/mAP50-95(B)": 0.4,
                }
            )

    monkeypatch.setattr(evaluation, "initialize_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        evaluation,
        "resolve_model_paths",
        lambda *args, **kwargs: (None, checkpoint),
    )
    monkeypatch.setattr(
        evaluation,
        "resolve_dataset_source",
        lambda config: SimpleNamespace(data=str(dataset / "data.yaml"), source=str(dataset)),
    )
    output_dir = tmp_path / "ultralytics-benchmark"
    result = evaluation.BenchmarkUltralyticsObjectDetection(
        BenchmarkObjectDetectionRequest(
            model_path=str(checkpoint),
            dataset_path=str(dataset),
            output_path=str(output_dir),
            confidence=0.001,
        )
    ).execute()

    assert result.metrics == pytest.approx(STANDARD_METRICS)
    assert calls[0]["project"] == str(output_dir.resolve().parent)
    assert calls[0]["name"] == output_dir.name
    assert json.loads((output_dir / "metrics.json").read_text()) == pytest.approx(
        STANDARD_METRICS
    )


def test_benchmark_provider_events_are_available_without_terminal_coupling(
    monkeypatch, tmp_path: Path
) -> None:
    from mlx.modes.object_detection.libreyolo import provider as provider_module

    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    output_dir = tmp_path / "benchmark"
    expected = ObjectDetectionBenchmarkResult(
        provider="libreyolo",
        model_path=str(checkpoint),
        dataset="dataset",
        split="test",
        metrics=STANDARD_METRICS,
        output_dir=output_dir,
        evaluation_backend="fake",
        native_metrics=STANDARD_METRICS,
    )

    class FakeBenchmark:
        def __init__(self, request):
            pass

        def execute(self):
            return expected

    from mlx.modes.object_detection.libreyolo import evaluation

    monkeypatch.setattr(evaluation, "BenchmarkLibreYOLOObjectDetection", FakeBenchmark)
    events = []
    result = provider_module.LibreYOLOProvider().benchmark(
        BenchmarkObjectDetectionRequest(model_path=str(checkpoint)),
        CallbackWorkflowReporter(events.append),
    )

    assert result is expected
    assert [event.level for event in events] == ["info", "success"]


def test_detection_runner_exposes_benchmark_with_research_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    captured = []
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    dataset = _dataset(tmp_path)
    expected = ObjectDetectionBenchmarkResult(
        provider="fake",
        model_path=str(checkpoint),
        dataset=str(dataset),
        split="test",
        metrics=STANDARD_METRICS,
        output_dir=tmp_path / "out",
        evaluation_backend="fake",
        native_metrics=STANDARD_METRICS,
    )

    class FakeCommand:
        def __init__(self, request, *, reporter=None):
            captured.append(request)

        def execute(self):
            return expected

    rendered = []
    monkeypatch.setattr(runner, "BenchmarkObjectDetectionModel", FakeCommand)
    monkeypatch.setattr(runner, "print_benchmark_result", rendered.append)

    result = runner.run_object_detection(
        {
            "action": "benchmark",
            "platform": "local",
            "provider": "fake",
            "model_path": str(checkpoint),
            "dataset_path": str(dataset),
            "_explicit_options": set(),
        }
    )

    assert result is expected
    assert captured[0].confidence == pytest.approx(0.001)
    assert captured[0].batch_size == 16
    assert (captured[0].height, captured[0].width) == (640, 640)
    assert rendered == [expected]


def test_sagemaker_stages_post_training_benchmark_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "work" / "trial" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    benchmark_dir = tmp_path / "work" / "trial" / "benchmark" / "val"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    benchmark = ObjectDetectionBenchmarkResult(
        provider="fake",
        model_path=str(checkpoint),
        dataset="dataset",
        split="val",
        metrics=STANDARD_METRICS,
        output_dir=benchmark_dir,
        evaluation_backend="fake",
        native_metrics=STANDARD_METRICS,
    )
    result = ObjectDetectionTrainingResult(
        training_result={},
        checkpoint_path=str(checkpoint),
        benchmark_result=benchmark,
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    command = RunSageMakerObjectDetectionTraining(
        model_dir=model_dir,
        work_dir=tmp_path / "work",
    )

    command._stage_model_artifacts(result, {"provider": "fake"})

    assert (model_dir / "best.pt").read_bytes() == b"checkpoint"
    assert (model_dir / "benchmark" / "metrics.json").is_file()
    assert (model_dir / "training-summary.json").is_file()
