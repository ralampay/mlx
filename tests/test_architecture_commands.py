from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx.cli import MODE_REGISTRY, build_parser
from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary
from mlx.modes.image_classification.requests import ImageClassificationRequest
from mlx.modes.object_detection.commands import (
    ConvertObjectDetectionModel,
    CreateObjectDetector,
    ListObjectDetectionModels,
    RunObjectDetectionStream,
    TrainObjectDetectionModel,
)
from mlx.modes.object_detection.models import Detection, DetectionResult
from mlx.modes.object_detection.requests import (
    ConvertObjectDetectionRequest,
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
    namespace = build_parser().parse_args(["--provider", "custom"])
    assert namespace.provider == "custom"


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
    assert [name for name, _ in provider.calls] == ["train", "create", "convert", "list"]


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


def test_architecture_documentation_is_canonical() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    architecture = (root / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "[Architecture](./ARCHITECTURE.md)" in readme
    assert "Review `ARCHITECTURE.md` for every code or configuration update" in agents
    assert "## Object-Detection Providers" in architecture
