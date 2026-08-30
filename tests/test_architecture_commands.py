from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx.cli import MODE_REGISTRY, build_parser, main
from mlx.cli_routing import resolve_mode_descriptor
from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary
from mlx.modes.image_classification.requests import ImageClassificationRequest
from mlx.modes.object_detection.commands import (
    BenchmarkObjectDetectionModel,
    ConvertObjectDetectionModel,
    CreateObjectDetector,
    FineTuneObjectDetectionModel,
    ListObjectDetectionModels,
    RunObjectDetectionStream,
    TrainObjectDetectionModel,
)
from mlx.modes.object_detection.models import Detection, DetectionResult
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    ConvertObjectDetectionRequest,
    FineTuneObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    TrainObjectDetectionRequest,
)
from mlx.modes.object_detection.ultralytics.results import (
    Detection as LegacyDetection,
)


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls = []

    def train(self, request, reporter):
        self.calls.append(("train", request))
        return "trained"

    def benchmark(self, request, reporter):
        self.calls.append(("benchmark", request))
        return "benchmarked"

    def create_detector(self, request):
        self.calls.append(("create", request))
        return FakeDetector()

    def convert(self, request, reporter):
        self.calls.append(("convert", request))
        return Path("model.onnx")

    def list_models(self, request, reporter):
        self.calls.append(("list", request))
        return [ModelParameterSummary("fake-model", 10)]


class FakeDetector:
    def __init__(self) -> None:
        self.frames = 0

    def predict(self, frame: np.ndarray) -> DetectionResult:
        self.frames += 1
        return DetectionResult(
            detections=(Detection((0, 0, 1, 1), 0.9, 0, "thing"),),
            names={0: "thing"},
        )


class FakeFrameSource:
    def __init__(self) -> None:
        self.frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeFrameSink:
    def __init__(self) -> None:
        self.frames = []
        self.closed = False

    def show(self, frame: np.ndarray) -> bool:
        self.frames.append(frame)
        return True

    def close(self) -> None:
        self.closed = True


def test_cli_routes_detection_through_neutral_runner_and_parses_provider() -> None:
    assert MODE_REGISTRY["object_detection"] == (
        "mlx.modes.object_detection.runner:run_object_detection"
    )
    namespace = build_parser().parse_args(
        ["--provider", "custom", "--profile", "mlx-training"]
    )
    assert namespace.provider == "custom"
    assert namespace.profile == "mlx-training"


def test_cli_parses_s3_dataset_staging_options() -> None:
    namespace = build_parser().parse_args(
        [
            "--dataset-s3-uri",
            "s3://datasets/training.zip",
            "--dataset-cache-dir",
            "/tmp/mlx-cache",
        ]
    )
    assert namespace.dataset_s3_uri == "s3://datasets/training.zip"
    assert namespace.dataset_cache_dir == "/tmp/mlx-cache"


def test_cli_parses_fine_tuning_model_s3_uri() -> None:
    namespace = build_parser().parse_args(
        ["--model-s3-uri", "s3://models/yolo9-best.pt"]
    )

    assert namespace.model_s3_uri == "s3://models/yolo9-best.pt"


def test_mode_descriptors_supply_defaults_and_alias_metadata() -> None:
    canonical = resolve_mode_descriptor("video_anomaly_detection")
    alias = resolve_mode_descriptor("video-anomaly-detection")

    assert canonical is alias
    assert canonical.default_action == "ls-models"
    assert "benchmark" in canonical.actions
    assert "best-model" in resolve_mode_descriptor("object_detection").actions
    assert "fine-tune" in resolve_mode_descriptor("object_detection").actions


def test_cli_applies_mode_default_before_dispatch_and_emits_clean_json(
    monkeypatch, capsys
) -> None:
    captured = {}

    def fake_resolver(_mode):
        def run(config):
            captured.update(config)
            return {"path": Path("artifact.json"), "ok": True}

        return run

    monkeypatch.setattr("mlx.cli._resolve_mode_runner", fake_resolver)

    assert main(["--mode", "image_classification", "--format", "json"]) == 0
    assert captured["action"] == "test"
    assert capsys.readouterr().out == '{"ok": true, "path": "artifact.json"}\n'


def test_typed_request_round_trips_legacy_extra_values() -> None:
    request = ImageClassificationRequest.from_config(
        {"model": "resnet18", "epochs": 3, "plugin_option": "kept"}
    )

    assert request.model == "resnet18"
    assert request.epochs == 3
    assert request.to_config()["plugin_option"] == "kept"


def test_detection_commands_delegate_to_injected_provider() -> None:
    provider = FakeProvider()

    assert TrainObjectDetectionModel(
        TrainObjectDetectionRequest(provider="fake"), provider=provider
    ).execute() == "trained"
    assert BenchmarkObjectDetectionModel(
        BenchmarkObjectDetectionRequest(provider="fake"), provider=provider
    ).execute() == "benchmarked"
    assert isinstance(
        CreateObjectDetector(ObjectDetectionRequest(provider="fake"), provider=provider).execute(),
        FakeDetector,
    )
    assert ConvertObjectDetectionModel(
        ConvertObjectDetectionRequest(provider="fake"), provider=provider
    ).execute() == Path("model.onnx")
    assert ListObjectDetectionModels(
        ListObjectDetectionModelsRequest(provider="fake"), provider=provider
    ).execute() == (ModelParameterSummary("fake-model", 10),)
    assert [name for name, _ in provider.calls] == [
        "train",
        "benchmark",
        "create",
        "convert",
        "list",
    ]


def test_fine_tune_command_requires_and_normalizes_checkpoint(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    checkpoint = tmp_path / "initial.pt"
    checkpoint.touch()

    result = FineTuneObjectDetectionModel(
        FineTuneObjectDetectionRequest(
            provider="fake",
            model_path=str(checkpoint),
        ),
        provider=provider,
    ).execute()

    assert result == "trained"
    assert provider.calls[0][1].model_path == str(checkpoint.resolve())

    with pytest.raises(MLXUserError, match="requires --model-path"):
        FineTuneObjectDetectionModel(
            FineTuneObjectDetectionRequest(provider="fake"),
            provider=provider,
        ).execute()


def test_unknown_detection_provider_is_user_facing() -> None:
    with pytest.raises(MLXUserError, match="Unsupported object-detection provider 'missing'"):
        CreateObjectDetector(ObjectDetectionRequest(provider="missing")).execute()


def test_stream_command_is_headless_and_closes_injected_ports() -> None:
    source = FakeFrameSource()
    sink = FakeFrameSink()
    events = []
    detector = FakeDetector()

    result = RunObjectDetectionStream(
        detector=detector,
        frame_source=source,
        frame_sink=sink,
        renderer=lambda frame, detection: frame + len(detection.detections),
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    assert result.frames_processed == 2
    assert result.stopped_by_user is False
    assert detector.frames == 2
    assert source.released is True
    assert sink.closed is True
    assert len(sink.frames) == 2
    assert events[-1].level == "success"


def test_legacy_ultralytics_detection_type_is_a_neutral_reexport() -> None:
    assert LegacyDetection is Detection


def test_benchmark_command_is_in_public_object_detection_api() -> None:
    from mlx.modes.object_detection import (
        BenchmarkObjectDetectionModel as PublicBenchmarkCommand,
    )

    assert PublicBenchmarkCommand is BenchmarkObjectDetectionModel


def test_architecture_documentation_is_canonical() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    architecture = (root / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "[Architecture](./ARCHITECTURE.md)" in readme
    assert "Review `ARCHITECTURE.md` for every code or configuration update" in agents
    assert "## Object-Detection Providers" in architecture
