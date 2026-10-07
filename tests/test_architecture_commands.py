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


def test_stream_reports_interval_and_average_processing_fps():
    source = FakeFrameSource()
    source.frames *= 2
    events = []
    times = iter([0.0, 0.25, 1.0, 1.25, 2.5])
    result = RunObjectDetectionStream(
        detector=FakeDetector(), frame_source=source, frame_sink=FakeFrameSink(),
        renderer=lambda frame, detection: frame,
        reporter=CallbackWorkflowReporter(events.append), clock=lambda: next(times),
    ).execute()

    progress = [event for event in events if event.level == "progress"]
    assert [event.current for event in progress] == [2, 4]
    assert [event.payload["fps"] for event in progress] == pytest.approx([2.0, 2 / 1.5])
    assert result.elapsed_seconds == 2.5
    assert result.average_fps == pytest.approx(4 / 2.5)
    assert events[-1].payload == {"event": "stream_summary", "average_fps": result.average_fps}
    assert source.released and not result.stopped_by_user


@pytest.mark.parametrize("frames,times,expected_elapsed,expected_fps", [
    (0, [10.0], 0.0, 0.0),
    (2, [10.0, 10.0, 10.0], 0.0, 0.0),
    (2, [10.0, 10.25, 10.5], 0.5, 4.0),
])
def test_short_and_empty_streams_report_finite_average_fps(frames, times, expected_elapsed, expected_fps):
    source = FakeFrameSource()
    source.frames = source.frames[:frames]
    ticks = iter(times)
    events = []
    result = RunObjectDetectionStream(
        detector=FakeDetector(), frame_source=source, frame_sink=FakeFrameSink(),
        renderer=lambda frame, detection: frame,
        reporter=CallbackWorkflowReporter(events.append), clock=lambda: next(ticks),
    ).execute()
    assert result.elapsed_seconds == expected_elapsed
    assert result.average_fps == expected_fps
    assert len(events) == 1
    assert source.released


def test_stopping_stream_includes_last_displayed_frame_in_fps():
    source = FakeFrameSource()
    sink = FakeFrameSink()
    sink.show = lambda frame: False
    times = iter([0.0, 2.0])
    result = RunObjectDetectionStream(
        detector=FakeDetector(), frame_source=source, frame_sink=sink,
        renderer=lambda frame, detection: frame, clock=lambda: next(times),
    ).execute()
    assert result.stopped_by_user
    assert result.frames_processed == 1
    assert result.average_fps == 0.5
    assert sink.closed and source.released


def test_detection_reporter_prints_live_and_average_fps(monkeypatch):
    from io import StringIO
    from rich.console import Console
    from mlx.core.commands import WorkflowEvent
    from mlx.modes.object_detection import presentation

    messages = []
    monkeypatch.setattr(presentation, "console", Console(file=StringIO(), force_terminal=False))
    monkeypatch.setattr(presentation, "print_info", messages.append)
    monkeypatch.setattr(presentation, "print_success", messages.append)
    reporter = presentation.RichWorkflowReporter()
    reporter.emit(WorkflowEvent("progress", "performance", payload={
        "event": "stream_fps", "fps": 12.345, "frames_processed": 20,
    }))
    reporter.emit(WorkflowEvent("success", "Completed.", payload={
        "event": "stream_summary", "average_fps": 10.125,
    }))
    assert messages == ["FPS: 12.35 (20 frames processed)", "Completed. Average FPS: 10.12"]


def test_detection_reporter_refreshes_one_stdout_line_and_clears_shorter_updates(monkeypatch):
    from io import StringIO
    from rich.console import Console
    from mlx.core.commands import WorkflowEvent
    from mlx.modes.object_detection import presentation

    output = StringIO()
    monkeypatch.setattr(presentation, "console", Console(file=output, force_terminal=True, width=100))
    monkeypatch.setattr(presentation, "print_success", lambda message: output.write(message + "\n"))
    reporter = presentation.RichWorkflowReporter()
    for fps in (100.0, 9.0):
        reporter.emit(WorkflowEvent("progress", "performance", payload={
            "event": "stream_fps", "fps": fps, "frames_processed": 20,
        }))
    assert output.getvalue() == (
        "\rFPS: 100.00 (20 frames processed)\rFPS: 9.00 (20 frames processed)  "
    )
    reporter.emit(WorkflowEvent("success", "Completed.", payload={
        "event": "stream_summary", "average_fps": 10.0,
    }))
    assert output.getvalue().endswith("\nCompleted. Average FPS: 10.00\n")


def test_legacy_ultralytics_detection_type_is_a_neutral_reexport() -> None:
    assert LegacyDetection is Detection


@pytest.mark.parametrize("failure_stage", ["read", "predict", "render", "show", "release", "close"])
def test_stream_failure_closes_both_ports_and_propagates_error(monkeypatch, failure_stage):
    source = FakeFrameSource()
    sink = FakeFrameSink()
    detector = FakeDetector()
    events = []
    failure = RuntimeError(f"{failure_stage} failed")

    def fail(*args):
        raise failure

    def renderer(frame, result):
        return frame

    if failure_stage == "render":
        renderer = fail
    else:
        target = {
            "read": source, "predict": detector, "show": sink,
            "release": source, "close": sink,
        }[failure_stage]
        monkeypatch.setattr(target, failure_stage, fail)

    with pytest.raises(RuntimeError) as caught:
        RunObjectDetectionStream(
            detector=detector,
            frame_source=source,
            frame_sink=sink,
            renderer=renderer,
            reporter=CallbackWorkflowReporter(events.append),
        ).execute()

    assert caught.value is failure
    assert source.released or failure_stage == "release"
    assert sink.closed or failure_stage == "close"
    assert events == []


@pytest.mark.parametrize("failure_stage", ["sink", "command"])
def test_detection_runner_cleans_up_failed_stream_setup(monkeypatch, failure_stage):
    from mlx.modes.object_detection import runner

    source = FakeFrameSource()
    sink = FakeFrameSink()
    failure = MLXUserError("Stream setup failed")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(runner.CreateObjectDetector, "execute", lambda self: FakeDetector())
    monkeypatch.setattr(runner, "OpenCVFrameSource", lambda **kwargs: source)
    monkeypatch.setattr(runner, "OpenCVFrameSink", lambda **kwargs: sink)
    monkeypatch.setattr(
        runner, "OpenCVFrameSink" if failure_stage == "sink" else "RunObjectDetectionStream", fail
    )

    with pytest.raises(MLXUserError) as caught:
        runner.run_object_detection({"action": "infer-camera"})

    assert caught.value is failure
    assert source.released
    assert sink.closed is (failure_stage == "command")


@pytest.mark.parametrize("output_format", ["rich", "json"])
def test_detection_runner_transfers_ports_to_stream_once(monkeypatch, output_format):
    from mlx.modes.object_detection import runner

    source = FakeFrameSource()
    sink = FakeFrameSink()
    cleanup = []
    monkeypatch.setattr(source, "release", lambda: cleanup.append("source"))
    monkeypatch.setattr(sink, "close", lambda: cleanup.append("sink"))
    monkeypatch.setattr(runner.CreateObjectDetector, "execute", lambda self: FakeDetector())
    monkeypatch.setattr(runner, "OpenCVFrameSource", lambda **kwargs: source)
    monkeypatch.setattr(runner, "OpenCVFrameSink", lambda **kwargs: sink)
    monkeypatch.setattr(runner, "NullFrameSink", lambda: sink)
    monkeypatch.setattr(runner, "annotate_detections", lambda frame, result: frame)

    result = runner.run_object_detection({
        "action": "infer-camera", "output_format": output_format,
    })

    assert result.frames_processed == 2
    assert cleanup == ["source", "sink"]


@pytest.mark.parametrize("arguments,expected", [
    ([], (640, 640)),
    (["--height", "320"], (320, 640)),
    (["--width=512"], (640, 512)),
    (["--height", "256", "--width", "256"], (256, 256)),
])
def test_detection_cli_uses_mode_defaults_and_preserves_explicit_dimensions(monkeypatch, arguments, expected):
    from mlx.cli import _build_config, _explicit_option_destinations
    from mlx.modes.object_detection import runner

    args = ["--mode", "object_detection", "--action", "infer-video", "--format", "json", *arguments]
    parser = build_parser()
    config = _build_config(parser.parse_args(args))
    config["_explicit_options"] = _explicit_option_destinations(parser, args)
    original_size = config["input_size"]
    received = []

    def create(command):
        received.append(command.request)
        return FakeDetector()

    monkeypatch.setattr(runner.CreateObjectDetector, "execute", create)
    monkeypatch.setattr(runner, "OpenCVFrameSource", lambda **kwargs: FakeFrameSource())
    monkeypatch.setattr(runner, "annotate_detections", lambda frame, result: frame)
    runner.run_object_detection(config)
    assert (received[0].height, received[0].width) == expected
    assert received[0].extras["input_size"] == (expected[1], expected[0])
    assert config["input_size"] == original_size


def test_python_detection_dimensions_and_other_mode_defaults_are_preserved(monkeypatch):
    from mlx.modes.object_detection import runner

    received = []
    monkeypatch.setattr(runner.CreateObjectDetector, "execute", lambda command: received.append(command.request) or FakeDetector())
    monkeypatch.setattr(runner, "OpenCVFrameSource", lambda **kwargs: FakeFrameSource())
    monkeypatch.setattr(runner, "annotate_detections", lambda frame, result: frame)
    runner.run_object_detection({"action": "infer-video", "output_format": "json", "height": 256, "width": 320})
    assert (received[0].height, received[0].width) == (256, 320)
    classification = build_parser().parse_args(["--mode", "image_classification"])
    assert (classification.height, classification.width) == (256, 256)


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
