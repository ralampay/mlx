from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mlx.cli import build_parser, CLIUsageError
from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.models import Detection, DetectionResult
from mlx.modes.object_detection.tracking.benchmark import BenchmarkTrackingDataset
from mlx.modes.object_detection.tracking.data import SequenceFrameSource, load_tracking_sequence, discover_tracking_sequences
from mlx.modes.object_detection.tracking.models import BoundingBox, TrackingFrameResult, TrackResult, TrackStatus
from mlx.modes.object_detection.tracking.preparation import MOTSequenceAdapter, PersonPathSequenceAdapter, PrepareTrackingBenchmarks
from mlx.modes.object_detection.tracking.presentation import TrackingTrajectoryRenderer
from mlx.modes.object_detection.tracking.requests import TrackingBenchmarkRequest, TrackingRequest
from mlx.modes.object_detection.tracking.video import TrackingVideoWriter


class Detector:
    def __init__(self):
        self.pixels = []

    def predict(self, frame):
        self.pixels.append(frame.copy())
        return DetectionResult((Detection((5, 5, 15, 25), .95, 0, "person"),), {0: "person"})


def sequence(root, name="sequence", split="val", count=3):
    path = root / "Example" / split / name
    (path / "img1").mkdir(parents=True)
    (path / "gt").mkdir()
    for index in range(count):
        image = np.random.default_rng(index).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(path / "img1" / f"{index + 1}.png"), image)
    (path / "gt/gt.txt").write_text("".join(f"{i+1},1,5,5,10,20,1,-1,-1,-1\n" for i in range(count)))
    write_json_atomic(path / "sequence.json", {
        "version": 1, "dataset": "Example", "split": split, "name": name,
        "kind": "images", "media": "img1", "ground_truth": "gt/gt.txt",
        "fps": 5, "width": 64, "height": 48, "frame_indices": list(range(count)),
    })
    return load_tracking_sequence(path / "sequence.json")


@pytest.mark.parametrize("args,expected", [([], True), (["--real-time-results"], True),
    (["--real-time-results", "True"], True), (["--real-time-results", "False"], False),
    (["--no-real-time-results"], False)])
def test_boolean_cli(args, expected):
    assert build_parser().parse_args(args).real_time_results is expected


def test_bad_boolean():
    with pytest.raises(CLIUsageError, match="True or False"):
        build_parser().parse_args(["--real-time-results", "perhaps"])


def test_visual_direct_parity_and_sequence_reset(tmp_path):
    dataset = tmp_path / "dataset"
    sequence(dataset, "one", count=12)
    sequence(dataset, "two", count=2)
    detectors = []
    results = []
    for visual in (False, True):
        detector = Detector()
        request = TrackingBenchmarkRequest(str(dataset), TrackingRequest(output_path=str(tmp_path / str(visual)), display=False), real_time_results=visual)
        result = BenchmarkTrackingDataset(request, detector=detector, renderer_factory=TrackingTrajectoryRenderer).execute()
        assert result.complete
        assert len(result.sequences) == 2
        for row in result.sequences:
            assert row["mota"] == 1
            assert row["idf1"] == 1
            tracks = row["result"].output_path.read_text().splitlines()
            assert all(line.split(",")[1] == "1" for line in tracks)
            assert bool(row["annotated_video"]) is visual
            if visual:
                capture = cv2.VideoCapture(row["annotated_video"])
                assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == row["frames_processed"]
                capture.release()
            else:
                assert not list(result.output_path.rglob("*.avi"))
        detectors.append(detector)
        results.append(result)
    assert len(detectors[0].pixels) == 14
    assert all(np.array_equal(a, b) for a, b in zip(detectors[0].pixels, detectors[1].pixels))
    assert (results[0].output_path / "summary.csv").exists()
    with pytest.raises(MLXUserError, match="already exist"):
        BenchmarkTrackingDataset(request, detector=Detector(), renderer_factory=TrackingTrajectoryRenderer).execute()


def test_cancel_skips_metrics_and_remaining_sequences(tmp_path):
    dataset = tmp_path / "dataset"
    sequence(dataset, "one")
    sequence(dataset, "two")

    class Display:
        def show(self, frame):
            return False

        def close(self):
            pass

    request = TrackingBenchmarkRequest(str(dataset), TrackingRequest(output_path=str(tmp_path / "results")))
    result = BenchmarkTrackingDataset(request, detector=Detector(), renderer_factory=TrackingTrajectoryRenderer, display_factory=Display).execute()
    assert not result.complete
    assert len(result.sequences) == 1
    assert result.sequences[0]["status"] == "cancelled"
    assert not list(result.output_path.rglob("metrics.json"))
    assert Path(result.sequences[0]["annotated_video"]).exists()


def test_overwrite_cancellation_removes_old_metrics(tmp_path):
    sequence(tmp_path / "dataset")
    request = TrackingBenchmarkRequest(str(tmp_path / "dataset"), TrackingRequest(output_path=str(tmp_path / "out"), display=False))
    BenchmarkTrackingDataset(request, detector=Detector(), renderer_factory=TrackingTrajectoryRenderer).execute()
    assert list((tmp_path / "out").rglob("metrics.json"))

    class Stop:
        def show(self, frame):
            return False

        def close(self):
            pass

    request = replace(request, tracking=replace(request.tracking, overwrite=True, display=True))
    result = BenchmarkTrackingDataset(request, detector=Detector(), renderer_factory=TrackingTrajectoryRenderer, display_factory=Stop).execute()
    assert not result.complete
    assert not list((tmp_path / "out").rglob("metrics.json"))


def test_inference_failure_cleans_partial_artifacts(tmp_path):
    sequence(tmp_path / "dataset")

    class FailingDetector:
        def predict(self, frame):
            raise MLXUserError("synthetic detector failure")

    request = TrackingBenchmarkRequest(str(tmp_path / "dataset"), TrackingRequest(output_path=str(tmp_path / "out"), display=False))
    with pytest.raises(MLXUserError, match="synthetic detector failure"):
        BenchmarkTrackingDataset(request, detector=FailingDetector(), renderer_factory=TrackingTrajectoryRenderer).execute()
    assert not list((tmp_path / "out").rglob("*.tmp"))
    assert not list((tmp_path / "out").rglob("*.partial.mp4"))
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert not report["complete"]
    assert report["sequences"][0]["status"] == "failed"


def test_duplicate_ground_truth_is_rejected_before_copy(tmp_path):
    from mlx.modes.object_detection.tracking.mot import MOTRecord
    command = PrepareTrackingBenchmarks(tmp_path, tmp_path / "out")
    record = MOTRecord(1, 1, 0, 0, 10, 10, 1)
    with pytest.raises(MLXUserError, match="Conflicting annotations"):
        command._publish({"dataset": "Example", "split": "train", "name": "one", "frame_indices": [0]}, [record, record], [])
    assert not (tmp_path / "out").exists()


def test_corrupt_input_and_manifest_are_actionable(tmp_path):
    seq = sequence(tmp_path)
    (seq.media / "2.png").unlink()
    with pytest.raises(MLXUserError, match="missing frame"):
        discover_tracking_sequences(tmp_path)
    manifest = json.loads((seq.root / "sequence.json").read_text())
    manifest["media"] = "../../../../outside"
    write_json_atomic(seq.root / "sequence.json", manifest)
    with pytest.raises(MLXUserError, match="inside"):
        load_tracking_sequence(seq.root / "sequence.json")


def test_mot_preparation_copies_and_maps_zero_ids(tmp_path):
    raw = tmp_path / "raw"
    source = raw / "DanceTrack/dataset/train1/dance"
    (source / "img1").mkdir(parents=True)
    (source / "gt").mkdir()
    cv2.imwrite(str(source / "img1/000001.jpg"), np.zeros((48, 64, 3), np.uint8))
    (source / "gt/gt.txt").write_text("1,0,5,5,10,20,1,1,1\n")
    output = tmp_path / "prepared"
    report = PrepareTrackingBenchmarks(raw, output).execute()
    assert report["ready"] == ["DanceTrack/train/dance"]
    seq = discover_tracking_sequences(output)[0]
    assert seq.ground_truth.read_text().startswith("1,1,")
    copied = next(seq.media.iterdir())
    assert copied.stat().st_ino != (source / "img1/000001.jpg").stat().st_ino
    assert not copied.is_symlink()
    assert any(r.get("dataset") == "HiEve" for r in report["unavailable"])
    with pytest.raises(MLXUserError, match="already exists"):
        PrepareTrackingBenchmarks(raw, output).execute()


def test_personpath_sample_alignment_and_conversion(tmp_path):
    video = tmp_path / "sample.avi"
    writer = TrackingVideoWriter(video, fps=25, width=64, height=48, lossless=True)
    for i in range(11):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.finalize()
    annotation = tmp_path / "anno.json"
    write_json_atomic(annotation, {"metadata": {"resolution": {"width": 64, "height": 48}},
        "entities": [{"id": 0, "blob": {"frame_idx": i}, "bb": [5, 5, 10, 20], "labels": {"person": 1}} for i in [0, 5, 10]]})
    manifest, records, _ = PersonPathSequenceAdapter().describe(video, annotation, "train")
    assert [r.frame_id for r in records] == [1, 2, 3]
    assert [r.track_id for r in records] == [1, 1, 1]
    assert manifest["frame_indices"] == [0, 5, 10]
    base = sequence(tmp_path / "dataset")
    source = SequenceFrameSource(replace(base, media=video, kind="video", frame_indices=(0, 5, 10)))
    try:
        assert [int(source.read()[1][0, 0, 0]) for _ in range(3)] == [0, 5, 10]
        assert source.read() == (False, None)
    finally:
        source.release()


def test_trails_are_bounded_and_expire():
    renderer = TrackingTrajectoryRenderer()
    frame = np.zeros((100, 100, 3), np.uint8)
    for i in range(1, 101):
        track = TrackResult(1, BoundingBox(10, 10, 20, 20), .9, 0, "person", TrackStatus.CONFIRMED, i, 0, i)
        renderer(frame, TrackingFrameResult(i, (track,)))
    assert len(renderer._history[1]) == 60
    assert not frame.any()
    renderer(frame, TrackingFrameResult(160, ()))
    assert not renderer._history


def test_runner_ignores_implicit_split_and_json_disables_display(monkeypatch):
    from mlx.modes.object_detection.tracking import runner
    captured = []

    class Command:
        def __init__(self, request, **kwargs):
            captured.append(request)

        def execute(self):
            return None

    monkeypatch.setattr(runner, "BenchmarkTrackingDataset", Command)
    runner.run_tracking({"action": "benchmark", "dataset_path": "data", "split": "test",
                         "_explicit_options": set(), "output_format": "json"})
    assert captured[0].split is None
    assert not captured[0].tracking.display
