from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx.cli import MODE_REGISTRY, build_parser
from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.models import Detection, DetectionResult
from mlx.modes.object_detection.streaming import FrameSourceMetadata
from mlx.modes.object_detection.tracking import (
    BoundingBox,
    ClassAwareTrackingRecord,
    ExportMOTFromClassAwareTracking,
    ExportTrackingReplay,
    TrackResult,
    TrackStatus,
    TrackingDetection,
    TrackingFrameResult,
    read_class_aware_tracking_file,
)
from mlx.modes.object_detection.tracking.algorithms import (
    ByteTrackAlgorithm,
    SortTrackingAlgorithm,
)
from mlx.modes.object_detection.tracking.evaluation import BenchmarkMOTTracking
from mlx.modes.object_detection.tracking.mot import (
    MOTRecord,
    MOTResultWriter,
    read_mot_file,
)
from mlx.modes.object_detection.tracking.presentation import annotate_tracks
from mlx.modes.object_detection.tracking.registry import (
    CreateTrackingAlgorithm,
    list_trackers,
    register_tracker,
)
from mlx.modes.object_detection.tracking.requests import TrackingRequest
from mlx.modes.object_detection.tracking.session import RunTrackingVideo


def make_detection(*, x: float = 0.0, confidence: float = 0.9, class_id: int = 0):
    return TrackingDetection(
        bounding_box=BoundingBox(x, 0, x + 10, 20),
        confidence=confidence,
        class_id=class_id,
        label="person",
    )


def test_cli_registers_tracking_mode_and_flags() -> None:
    assert MODE_REGISTRY["track"] == (
        "mlx.modes.object_detection.tracking.runner:run_tracking"
    )
    namespace = build_parser().parse_args(
        [
            "--mode",
            "track",
            "--tracker",
            "sort",
            "--ground-truth",
            "gt.txt",
            "--track-class-id",
            "0",
            "--track-class-id",
            "2",
            "--tracking-jsonl",
            "tracks.jsonl",
        ]
    )
    assert namespace.tracker == "sort"
    assert namespace.ground_truth == "gt.txt"
    assert namespace.track_class_ids == [0, 2]
    assert namespace.tracking_jsonl == "tracks.jsonl"


def test_tracking_request_reads_no_display_flag() -> None:
    namespace = build_parser().parse_args(["--mode", "track", "--no-display"])

    request = TrackingRequest.from_config(vars(namespace))

    assert request.display is False


def test_sort_preserves_identity_and_expires_lost_tracks() -> None:
    tracker = SortTrackingAlgorithm(min_hits=1, max_age=1, iou_threshold=0.2)

    first = tracker.update(frame_index=1, detections=(make_detection(),))
    second = tracker.update(frame_index=2, detections=(make_detection(x=1),))
    lost = tracker.update(frame_index=3, detections=())
    expired = tracker.update(frame_index=4, detections=())

    assert first.tracks[0].track_id == second.tracks[0].track_id == 1
    assert second.tracks[0].status is TrackStatus.CONFIRMED
    assert lost.tracks[0].status is TrackStatus.LOST
    assert expired.tracks == ()


def test_trackers_reject_zero_iou_threshold() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        SortTrackingAlgorithm(iou_threshold=0)
    with pytest.raises(ValueError, match="greater than 0"):
        ByteTrackAlgorithm(iou_threshold=0)


def test_sort_association_is_class_aware() -> None:
    tracker = SortTrackingAlgorithm(min_hits=1)
    tracker.update(frame_index=1, detections=(make_detection(class_id=0),))

    result = tracker.update(frame_index=2, detections=(make_detection(class_id=1),))

    assert {track.track_id for track in result.tracks} == {1, 2}
    assert next(track for track in result.tracks if track.track_id == 1).status is TrackStatus.LOST


def test_sort_requires_consecutive_hits_before_confirming_tentative_track() -> None:
    tracker = SortTrackingAlgorithm(min_hits=3, max_age=3)
    tracker.update(frame_index=1, detections=(make_detection(),))
    tracker.update(frame_index=2, detections=())
    recovered = tracker.update(frame_index=3, detections=(make_detection(),))
    confirmed = tracker.update(frame_index=4, detections=(make_detection(),))

    assert recovered.tracks[0].status is TrackStatus.TENTATIVE
    assert confirmed.tracks[0].status is TrackStatus.TENTATIVE
    confirmed = tracker.update(frame_index=5, detections=(make_detection(),))
    assert confirmed.tracks[0].status is TrackStatus.CONFIRMED


def test_bytetrack_uses_low_confidence_detection_to_recover_track() -> None:
    tracker = ByteTrackAlgorithm(min_hits=1, max_age=2)
    first = tracker.update(frame_index=1, detections=(make_detection(confidence=0.9),))

    recovered = tracker.update(
        frame_index=2,
        detections=(make_detection(x=1, confidence=0.2),),
    )

    assert first.tracks[0].track_id == recovered.tracks[0].track_id == 1
    assert recovered.tracks[0].status is TrackStatus.CONFIRMED
    assert recovered.tracks[0].confidence == pytest.approx(0.2)


def test_bytetrack_does_not_create_tracks_from_low_confidence_detections() -> None:
    result = ByteTrackAlgorithm().update(
        frame_index=1,
        detections=(make_detection(confidence=0.2),),
    )

    assert result.tracks == ()


def test_tracker_registry_loads_builtins_and_external_class(monkeypatch) -> None:
    assert list_trackers() == ("bytetrack", "sort")
    assert isinstance(
        CreateTrackingAlgorithm(tracker="sort", options={"min_hits": 1}).execute(),
        SortTrackingAlgorithm,
    )

    class ExternalTracker:
        def __init__(self, *, marker: int = 0) -> None:
            self.marker = marker

        def update(self, *, frame_index, detections, frame=None):
            return TrackingFrameResult(frame_index=frame_index, tracks=())

        def reset(self) -> None:
            return None

    module = ModuleType("fake_tracker_module")
    module.ExternalTracker = ExternalTracker
    monkeypatch.setitem(sys.modules, "fake_tracker_module", module)

    custom_registry = register_tracker(
        "external",
        "fake_tracker_module:ExternalTracker",
    )
    loaded = CreateTrackingAlgorithm(
        tracker="external",
        options={"marker": 7},
        registry=custom_registry,
    ).execute()

    assert loaded.marker == 7
    assert "external" in custom_registry.names()
    assert list_trackers() == ("bytetrack", "sort")
    with pytest.raises(TypeError):
        custom_registry.entries["another"] = "fake_tracker_module:ExternalTracker"


def test_tracker_registry_reads_json_and_reports_invalid_options(tmp_path: Path) -> None:
    config = tmp_path / "tracker.json"
    config.write_text(json.dumps({"min_hits": 1, "max_age": 4}), encoding="utf-8")
    tracker = CreateTrackingAlgorithm(
        tracker="sort",
        config_path=str(config),
    ).execute()
    assert tracker.min_hits == 1
    assert tracker.max_age == 4

    config.write_text(json.dumps({"unknown": True}), encoding="utf-8")
    with pytest.raises(MLXUserError, match="rejected its configuration"):
        CreateTrackingAlgorithm(tracker="sort", config_path=str(config)).execute()


def test_tracker_registry_translates_missing_external_dependency(monkeypatch) -> None:
    class DependencyTracker:
        def __init__(self) -> None:
            raise ImportError("optional-reid")

        def update(self, *, frame_index, detections, frame=None):
            return TrackingFrameResult(frame_index=frame_index, tracks=())

        def reset(self) -> None:
            return None

    module = ModuleType("dependency_tracker_module")
    module.DependencyTracker = DependencyTracker
    monkeypatch.setitem(sys.modules, "dependency_tracker_module", module)

    with pytest.raises(MLXUserError, match="missing a required dependency"):
        CreateTrackingAlgorithm(
            tracker="dependency_tracker_module:DependencyTracker"
        ).execute()


def test_mot_writer_uses_ten_columns_and_omits_lost_tracks(tmp_path: Path) -> None:
    tracker = SortTrackingAlgorithm(min_hits=1)
    writer = MOTResultWriter(output_dir=tmp_path)
    writer.start()
    writer.write(tracker.update(frame_index=1, detections=(make_detection(),)))
    writer.write(tracker.update(frame_index=2, detections=()))
    path = writer.finalize()

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert len(rows[0].split(",")) == 10
    assert rows[0].endswith(",-1,-1,-1")


def test_class_aware_records_export_standard_mot_with_optional_class_filter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tracks.jsonl"
    records = (
        ClassAwareTrackingRecord(
            frame_id=1,
            track_id=1,
            class_id=0,
            label="person",
            bounding_box=BoundingBox(1, 2, 11, 22),
            confidence=0.9,
        ),
        ClassAwareTrackingRecord(
            frame_id=1,
            track_id=2,
            class_id=2,
            label="car",
            bounding_box=BoundingBox(20, 20, 25, 25),
            confidence=0.8,
        ),
    )
    source.write_text(
        "".join(record.to_json_line() + "\n" for record in records),
        encoding="utf-8",
    )

    result = ExportMOTFromClassAwareTracking(
        source_path=source,
        output_dir=tmp_path / "person-mot",
        class_ids=(0,),
    ).execute()

    mot_records = read_mot_file(result.output_path)
    assert result.rows_written == 1
    assert result.source_rows == 2
    assert len(result.output_path.read_text(encoding="utf-8").split(",")) == 10
    assert [(record.frame_id, record.track_id) for record in mot_records] == [(1, 1)]

    all_result = ExportMOTFromClassAwareTracking(
        source_path=source,
        output_dir=tmp_path / "all-mot",
    ).execute()
    assert all_result.output_path.read_text(encoding="utf-8") == "".join(
        record.to_mot_record().to_line() + "\n" for record in records
    )


def test_class_aware_reader_rejects_invalid_schema_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tracks.jsonl"
    source.write_text(
        '{"schema_version":"unknown","frame_id":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(MLXUserError, match="unsupported schema_version"):
        read_class_aware_tracking_file(source)

    invalid_id = ClassAwareTrackingRecord(
        frame_id=1,
        track_id=1,
        class_id=0,
        label=None,
        bounding_box=BoundingBox(0, 0, 10, 10),
        confidence=1,
    ).to_dict()
    invalid_id["frame_id"] = 1.5
    source.write_text(json.dumps(invalid_id) + "\n", encoding="utf-8")
    with pytest.raises(MLXUserError, match="frame_id must be a JSON integer"):
        read_class_aware_tracking_file(source)

    record = ClassAwareTrackingRecord(
        frame_id=1,
        track_id=1,
        class_id=0,
        label=None,
        bounding_box=BoundingBox(0, 0, 10, 10),
        confidence=1,
    )
    source.write_text(
        f"{record.to_json_line()}\n{record.to_json_line()}\n",
        encoding="utf-8",
    )
    with pytest.raises(MLXUserError, match="appears more than once"):
        read_class_aware_tracking_file(source)


def test_mot_reader_rejects_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("1,2,3\n", encoding="utf-8")

    with pytest.raises(MLXUserError, match="expected 10"):
        read_mot_file(path)


def test_mot_reader_rejects_duplicate_track_ids_in_frame(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.txt"
    row = MOTRecord(1, 2, 0, 0, 10, 10, 1).to_line()
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(MLXUserError, match="more than once"):
        read_mot_file(path)


def test_mot_benchmark_reports_perfect_tracking(tmp_path: Path) -> None:
    row = MOTRecord(1, 1, 10, 20, 30, 40, 1)
    ground_truth = tmp_path / "gt.txt"
    predictions = tmp_path / "tracks.txt"
    metrics = tmp_path / "nested" / "metrics.json"
    ground_truth.write_text(row.to_line() + "\n", encoding="utf-8")
    predictions.write_text(row.to_line() + "\n", encoding="utf-8")

    result = BenchmarkMOTTracking(
        ground_truth_path=ground_truth,
        predictions_path=predictions,
        output_path=metrics,
        processed_frames=3,
    ).execute()

    assert result.mota == pytest.approx(1.0)
    assert result.motp == pytest.approx(1.0)
    assert result.idf1 == pytest.approx(1.0)
    assert result.false_positives == result.misses == result.id_switches == 0
    assert result.frames == 3
    assert json.loads(metrics.read_text(encoding="utf-8"))["mota"] == pytest.approx(1.0)


class FakeDetector:
    def predict(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult(
            detections=(
                Detection((1, 2, 11, 22), 0.9, 0, "person"),
                Detection((20, 20, 25, 25), 0.9, 2, "car"),
            ),
            names={0: "person", 2: "car"},
        )


class FailingDetector:
    def predict(self, frame: np.ndarray) -> DetectionResult:
        raise MLXUserError("Synthetic inference failure.")


class FakeFrameSource:
    def __init__(self) -> None:
        self.frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(2)]
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True

    def metadata(self) -> FrameSourceMetadata:
        return FrameSourceMetadata(width=40, height=30, fps=12.5, frame_count=2)


class EmptyFrameSource(FakeFrameSource):
    def __init__(self) -> None:
        super().__init__()
        self.frames = []


class FailingReleaseFrameSource(FakeFrameSource):
    def release(self) -> None:
        self.released = True
        raise RuntimeError("Synthetic release failure.")


class FailingMetadataFrameSource(FakeFrameSource):
    def metadata(self) -> FrameSourceMetadata:
        raise RuntimeError("Synthetic metadata failure.")


class FakeFrameSink:
    def __init__(self, *, stop_after: int | None = None) -> None:
        self.stop_after = stop_after
        self.frames: list[np.ndarray] = []
        self.closed = False

    def show(self, frame: np.ndarray) -> bool:
        self.frames.append(frame)
        return self.stop_after is None or len(self.frames) < self.stop_after

    def close(self) -> None:
        self.closed = True


def test_track_renderer_draws_current_observations_without_mutating_frame() -> None:
    frame = np.zeros((100, 220, 3), dtype=np.uint8)
    result = TrackingFrameResult(
        frame_index=2,
        tracks=(
            TrackResult(
                track_id=1,
                bounding_box=BoundingBox(20, 40, 60, 80),
                confidence=0.9,
                class_id=0,
                label="person",
                status=TrackStatus.CONFIRMED,
                hits=2,
                missing_frames=0,
                last_seen_frame=2,
            ),
            TrackResult(
                track_id=2,
                bounding_box=BoundingBox(140, 40, 180, 80),
                confidence=0.8,
                class_id=0,
                label="person",
                status=TrackStatus.LOST,
                hits=1,
                missing_frames=1,
                last_seen_frame=1,
            ),
        ),
    )

    annotated = annotate_tracks(frame, result)

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0
    assert np.any(annotated[40, 20] != 0)
    assert np.all(annotated[80, 140] == 0)


def test_tracking_video_command_requires_sink_and_renderer_together() -> None:
    with pytest.raises(ValueError, match="both a frame sink and renderer"):
        RunTrackingVideo(TrackingRequest(), frame_sink=FakeFrameSink())


def test_tracking_video_command_streams_results_and_filters_classes(tmp_path: Path) -> None:
    source = FakeFrameSource()
    events = []
    result = RunTrackingVideo(
        TrackingRequest(
            output_path=str(tmp_path / "output"),
            track_class_ids=(0,),
        ),
        detector=FakeDetector(),
        algorithm=ByteTrackAlgorithm(min_hits=1),
        frame_source=source,
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    records = read_mot_file(result.output_path)
    class_records = read_class_aware_tracking_file(result.class_aware_output_path)
    assert result.frames_processed == 2
    assert result.tracks_written == 2
    assert {record.track_id for record in records} == {1}
    assert result.class_aware_output_path == tmp_path / "output" / "tracks.jsonl"
    assert [(record.class_id, record.label) for record in class_records] == [
        (0, "person"),
        (0, "person"),
    ]
    assert source.released is True
    assert result.replay_data_path == tmp_path / "output" / "replay.json"
    assert result.replay_html_path == tmp_path / "output" / "replay.html"
    replay = json.loads(result.replay_data_path.read_text(encoding="utf-8"))
    assert replay["schema_version"] == "mlx.tracking.replay/v1"
    assert replay["canvas"] == {"width": 40, "height": 30}
    assert replay["fps"] == pytest.approx(12.5)
    assert replay["predictions"]["record_count"] == 2
    assert {
        (row["class_id"], row["label"])
        for row in replay["predictions"]["records"]
    } == {(0, "person")}
    assert events[-1].level == "success"


def test_tracking_video_command_falls_back_when_metadata_is_unavailable(
    tmp_path: Path,
) -> None:
    events = []

    result = RunTrackingVideo(
        TrackingRequest(output_path=str(tmp_path / "output")),
        detector=FakeDetector(),
        algorithm=ByteTrackAlgorithm(min_hits=1),
        frame_source=FailingMetadataFrameSource(),
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    replay = json.loads(result.replay_data_path.read_text(encoding="utf-8"))
    assert replay["fps"] == pytest.approx(30.0)
    assert replay["canvas"] == {"width": 40, "height": 30}
    assert any(
        event.level == "warning" and "optional frame-source metadata" in event.message
        for event in events
    )


def test_tracking_video_command_displays_results_and_stops_cleanly(
    tmp_path: Path,
) -> None:
    source = FakeFrameSource()
    sink = FakeFrameSink(stop_after=1)
    ground_truth = tmp_path / "gt.txt"
    ground_truth.write_text(
        "\n".join(
            (
                MOTRecord(1, 1, 1, 2, 10, 20, 1).to_line(),
                MOTRecord(2, 1, 1, 2, 10, 20, 1).to_line(),
            )
        ),
        encoding="utf-8",
    )
    events = []

    result = RunTrackingVideo(
        TrackingRequest(
            output_path=str(tmp_path / "output"),
            ground_truth=str(ground_truth),
        ),
        detector=FakeDetector(),
        algorithm=ByteTrackAlgorithm(min_hits=1),
        frame_source=source,
        frame_sink=sink,
        renderer=lambda frame, tracking_result: frame + len(tracking_result.tracks),
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    assert result.frames_processed == 1
    assert result.stopped_by_user is True
    assert result.benchmark is None
    assert result.metrics_path is None
    assert result.replay_html_path is not None
    replay = json.loads(result.replay_data_path.read_text(encoding="utf-8"))
    assert replay["frame_count"] == 1
    assert {row["frame_id"] for row in replay["ground_truth"]["records"]} == {1}
    assert len(sink.frames) == 1
    assert np.any(sink.frames[0] != 0)
    assert sink.closed is True
    assert source.released is True
    assert any(
        event.level == "warning" and "skipped benchmarking" in event.message
        for event in events
    )


def test_export_tracking_replay_is_portable_and_infers_canvas(tmp_path: Path) -> None:
    predictions = tmp_path / "tracks.txt"
    ground_truth = tmp_path / "gt.txt"
    predictions.write_text(
        MOTRecord(2, 4, 10, 20, 30, 40, 0.75).to_line() + "\n",
        encoding="utf-8",
    )
    ground_truth.write_text(
        MOTRecord(2, 8, 12, 22, 35, 45, 1).to_line() + "\n",
        encoding="utf-8",
    )

    result = ExportTrackingReplay(
        predictions_path=predictions,
        ground_truth_path=ground_truth,
        output_dir=tmp_path / "replay",
        fps=25,
        metrics={"mota": 0.5},
        run_metadata={"provider": "test", "tracker": "example"},
    ).execute()

    payload = json.loads(result.data_path.read_text(encoding="utf-8"))
    html = result.html_path.read_text(encoding="utf-8")
    assert payload["frame_count"] == 2
    assert payload["canvas"] == {"width": 47, "height": 67}
    assert payload["predictions"]["track_count"] == 1
    assert payload["ground_truth"]["track_count"] == 1
    assert payload["metrics"] == {"mota": 0.5}
    assert "mlx.tracking.replay/v1" in html
    assert "http://" not in html and "https://" not in html

    fixed_result = ExportTrackingReplay(
        predictions_path=predictions,
        output_dir=tmp_path / "fixed-replay",
        frame_width=40,
        frame_height=50,
        fps=25,
    ).execute()
    fixed_payload = json.loads(fixed_result.data_path.read_text(encoding="utf-8"))
    assert fixed_payload["canvas"] == {"width": 40, "height": 50}

    with pytest.raises(MLXUserError, match="replay artifact already exists"):
        ExportTrackingReplay(
            predictions_path=predictions,
            output_dir=tmp_path / "replay",
        ).execute()


def test_export_tracking_replay_rejects_mismatched_class_metadata(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "tracks.txt"
    class_aware = tmp_path / "tracks.jsonl"
    predictions.write_text(
        MOTRecord(1, 1, 0, 0, 10, 10, 0.9).to_line() + "\n",
        encoding="utf-8",
    )
    class_aware.write_text(
        ClassAwareTrackingRecord(
            frame_id=1,
            track_id=2,
            class_id=0,
            label="person",
            bounding_box=BoundingBox(0, 0, 10, 10),
            confidence=0.9,
        ).to_json_line()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MLXUserError, match="same frame/track rows"):
        ExportTrackingReplay(
            predictions_path=predictions,
            class_aware_path=class_aware,
            output_dir=tmp_path / "replay",
        ).execute()


def test_tracking_runner_exports_mot_from_class_aware_jsonl(tmp_path: Path) -> None:
    from mlx.modes.object_detection.tracking import runner

    source = tmp_path / "tracks.jsonl"
    record = ClassAwareTrackingRecord(
        frame_id=1,
        track_id=1,
        class_id=3,
        label="bike",
        bounding_box=BoundingBox(1, 2, 4, 6),
        confidence=0.7,
    )
    source.write_text(record.to_json_line() + "\n", encoding="utf-8")

    result = runner.run_tracking(
        {
            "action": "export-mot",
            "tracking_jsonl": str(source),
            "output_path": str(tmp_path / "mot"),
            "track_class_ids": [3],
        }
    )

    assert result.output_path == tmp_path / "mot" / "tracks.txt"
    assert result.rows_written == 1


def test_tracking_runner_wires_visualization_from_display_flag(monkeypatch) -> None:
    from mlx.modes.object_detection.tracking import runner

    captured = {}
    sink = FakeFrameSink()

    class FakeOpenCVFrameSink:
        def __new__(cls, **kwargs):
            captured["sink_options"] = kwargs
            return sink

    class FakeRunTrackingVideo:
        def __init__(self, request, **kwargs):
            captured["request"] = request
            captured.update(kwargs)

        def execute(self):
            return SimpleNamespace(benchmark=None)

    monkeypatch.setattr(runner, "OpenCVFrameSink", FakeOpenCVFrameSink)
    monkeypatch.setattr(runner, "RunTrackingVideo", FakeRunTrackingVideo)

    runner.run_tracking({"action": "run", "display": True})

    assert captured["frame_sink"] is sink
    assert captured["renderer"] is annotate_tracks
    assert captured["sink_options"] == {"title": "MLX Tracking", "delay_ms": 10}

    captured.clear()
    runner.run_tracking({"action": "run", "display": False})

    assert captured["frame_sink"] is None
    assert captured["renderer"] is None
    assert "sink_options" not in captured


def test_tracking_video_command_requires_clear_inputs() -> None:
    with pytest.raises(MLXUserError, match="--file-path"):
        RunTrackingVideo(TrackingRequest()).execute()


def test_tracking_video_command_removes_temporary_output_for_empty_video(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "output"
    source = EmptyFrameSource()

    with pytest.raises(MLXUserError, match="did not produce any readable frames"):
        RunTrackingVideo(
            TrackingRequest(file_path=str(video), output_path=str(output)),
            detector=FakeDetector(),
            algorithm=ByteTrackAlgorithm(),
            frame_source=source,
        ).execute()

    assert source.released is True
    assert not (output / "tracks.txt").exists()
    assert not (output / ".tracks.txt.tmp").exists()
    assert not (output / "tracks.jsonl").exists()
    assert not (output / ".tracks.jsonl.tmp").exists()


def test_tracking_video_command_rejects_existing_metric_before_processing(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    ground_truth = tmp_path / "gt.txt"
    ground_truth.write_text(MOTRecord(1, 1, 0, 0, 1, 1, 1).to_line(), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(MLXUserError, match="Tracking artifact already exists"):
        RunTrackingVideo(
            TrackingRequest(
                file_path=str(video),
                output_path=str(output),
                ground_truth=str(ground_truth),
            ),
            detector=FakeDetector(),
            algorithm=ByteTrackAlgorithm(),
            frame_source=FakeFrameSource(),
        ).execute()


def test_tracking_video_command_preserves_primary_error_during_cleanup(
    tmp_path: Path,
) -> None:
    source = FailingReleaseFrameSource()
    events = []

    with pytest.raises(MLXUserError, match="Synthetic inference failure"):
        RunTrackingVideo(
            TrackingRequest(output_path=str(tmp_path / "output")),
            detector=FailingDetector(),
            algorithm=ByteTrackAlgorithm(),
            frame_source=source,
            reporter=CallbackWorkflowReporter(events.append),
        ).execute()

    assert source.released is True
    assert any(
        event.level == "warning" and "release frame source" in event.message
        for event in events
    )
    assert not (tmp_path / "output" / ".tracks.txt.tmp").exists()


def test_tracking_video_command_validates_ground_truth_before_inference(
    tmp_path: Path,
) -> None:
    ground_truth = tmp_path / "gt.txt"
    ground_truth.write_text("not,a,mot,row\n", encoding="utf-8")
    detector = FakeDetector()

    with pytest.raises(MLXUserError, match="expected 10"):
        RunTrackingVideo(
            TrackingRequest(
                output_path=str(tmp_path / "output"),
                ground_truth=str(ground_truth),
            ),
            detector=detector,
            algorithm=ByteTrackAlgorithm(),
            frame_source=FakeFrameSource(),
        ).execute()
