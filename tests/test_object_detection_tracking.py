from __future__ import annotations

import gc
import math
import weakref
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking import (
    BoundingBox,
    RunTrackByDetectionCommand,
    TrackResult,
    TrackStatus,
    TrackingAlgorithm,
    TrackingDetection,
    TrackingFrameResult,
)
from mlx.modes.object_detection.tracking.algorithms import DetectionAsTrackAlgorithm
from mlx.modes.object_detection.tracking.models import AssociationResult
from mlx.modes.object_detection.ultralytics.results import Detection, DetectionResult
from mlx.modes.object_detection.ultralytics.tracking_adapter import (
    RunObjectDetectionTrackingCommand,
    to_tracking_detections,
)


def make_detection(
    *,
    x_offset: float = 0.0,
    confidence: float = 0.8,
    class_id: int = 2,
    label: str | None = "car",
) -> TrackingDetection:
    return TrackingDetection(
        bounding_box=BoundingBox(x_offset, 2.0, x_offset + 10.0, 22.0),
        confidence=confidence,
        class_id=class_id,
        label=label,
    )


def test_bounding_box_geometry_and_float_storage() -> None:
    bounding_box = BoundingBox(1, 2, 6, 10)

    assert bounding_box.as_xyxy() == (1.0, 2.0, 6.0, 10.0)
    assert bounding_box.width == 5.0
    assert bounding_box.height == 8.0
    assert bounding_box.area == 40.0
    assert all(isinstance(coordinate, float) for coordinate in bounding_box.as_xyxy())


@pytest.mark.parametrize(
    "coordinates",
    [
        (2.0, 0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0, 1.0),
    ],
)
def test_bounding_box_rejects_inverted_coordinates(coordinates) -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        BoundingBox(*coordinates)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_bounding_box_rejects_non_finite_coordinates(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BoundingBox(0.0, 0.0, invalid, 1.0)


def test_tracking_detection_rejects_non_finite_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be a finite number"):
        make_detection(confidence=math.nan)


def test_association_result_documents_and_freezes_match_convention() -> None:
    result = AssociationResult(
        matches=((12, 3),),
        unmatched_track_ids=(5,),
        unmatched_detection_indices=(1,),
    )

    assert result.matches == ((12, 3),)
    with pytest.raises(FrozenInstanceError):
        result.matches = ()


def test_placeholder_creates_confirmed_tracks_with_monotonic_ids() -> None:
    algorithm = DetectionAsTrackAlgorithm()
    detections = (make_detection(), make_detection(x_offset=20.0))

    first = algorithm.update(frame_index=1, detections=detections)
    second = algorithm.update(frame_index=2, detections=(make_detection(),))

    assert [track.track_id for track in first.tracks] == [1, 2]
    assert [track.track_id for track in second.tracks] == [3]
    assert all(track.status is TrackStatus.CONFIRMED for track in first.tracks)
    assert first.tracks[0].bounding_box is detections[0].bounding_box
    assert first.tracks[0].last_seen_frame == 1


def test_placeholder_does_not_retain_frames() -> None:
    algorithm = DetectionAsTrackAlgorithm()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_reference = weakref.ref(frame)

    algorithm.update(frame_index=1, detections=(), frame=frame)
    del frame
    gc.collect()

    assert frame_reference() is None


def test_placeholder_reset_restarts_track_ids() -> None:
    algorithm = DetectionAsTrackAlgorithm()
    algorithm.update(frame_index=1, detections=(make_detection(),))

    algorithm.reset()
    result = algorithm.update(frame_index=1, detections=(make_detection(),))

    assert result.tracks[0].track_id == 1


class RecordingAlgorithm:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object, object]] = []
        self.reset_calls = 0

    def update(
        self,
        *,
        frame_index: int,
        detections,
        frame=None,
    ) -> TrackingFrameResult:
        self.calls.append((frame_index, detections, frame))
        return TrackingFrameResult(frame_index=frame_index, tracks=())

    def reset(self) -> None:
        self.reset_calls += 1


def test_command_increments_and_delegates_inputs_unchanged() -> None:
    algorithm = RecordingAlgorithm()
    command = RunTrackByDetectionCommand(algorithm=algorithm)
    detections = [make_detection()]
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    first = command.execute(detections=detections, frame=frame)
    second = command.execute(detections=(), frame=None)

    assert first.frame_index == 1
    assert second.frame_index == 2
    assert command.frame_index == 2
    assert algorithm.calls[0] == (1, detections, frame)
    assert algorithm.calls[0][1] is detections
    assert algorithm.calls[0][2] is frame
    assert algorithm.calls[1] == (2, (), None)


def test_command_reset_resets_command_and_algorithm() -> None:
    algorithm = RecordingAlgorithm()
    command = RunTrackByDetectionCommand(algorithm=algorithm)
    command.execute(detections=())

    command.reset()

    assert command.frame_index == 0
    assert algorithm.reset_calls == 1
    assert command.execute(detections=()).frame_index == 1


def test_command_with_placeholder_does_not_retain_frame() -> None:
    command = RunTrackByDetectionCommand(algorithm=DetectionAsTrackAlgorithm())
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_reference = weakref.ref(frame)

    command.execute(detections=(), frame=frame)
    del frame
    gc.collect()

    assert frame_reference() is None


@pytest.mark.parametrize(
    "frame",
    [
        np.empty((0, 2, 3)),
        np.empty((2, 0)),
        np.empty((2,)),
        np.empty((1, 2, 3, 4)),
    ],
)
def test_command_rejects_empty_or_invalid_frame_shapes(frame: np.ndarray) -> None:
    command = RunTrackByDetectionCommand(algorithm=DetectionAsTrackAlgorithm())

    with pytest.raises(MLXUserError, match="Tracking frame"):
        command.execute(detections=(), frame=frame)

    assert command.frame_index == 0


def test_command_and_results_use_immutable_public_snapshots() -> None:
    command = RunTrackByDetectionCommand(algorithm=DetectionAsTrackAlgorithm())
    result = command.execute(detections=(make_detection(),))

    assert isinstance(result.tracks, tuple)
    with pytest.raises(FrozenInstanceError):
        result.frame_index = 10
    with pytest.raises(FrozenInstanceError):
        result.tracks[0].hits = 50
    assert not hasattr(result.tracks[0], "created_at_frame")


def test_tracking_frame_result_rejects_negative_index_and_duplicate_ids() -> None:
    track = TrackResult(
        track_id=1,
        bounding_box=BoundingBox(0, 0, 1, 1),
        confidence=0.5,
        class_id=0,
        label=None,
        status=TrackStatus.TENTATIVE,
        hits=1,
        missing_frames=0,
        last_seen_frame=0,
    )

    with pytest.raises(ValueError, match="zero or greater"):
        TrackingFrameResult(frame_index=-1, tracks=())
    with pytest.raises(ValueError, match="duplicate track IDs"):
        TrackingFrameResult(frame_index=0, tracks=(track, track))


def test_tracking_algorithm_protocol_is_runtime_checkable() -> None:
    assert isinstance(DetectionAsTrackAlgorithm(), TrackingAlgorithm)


def test_ultralytics_boundary_converts_normalized_detections() -> None:
    source = Detection(
        xyxy=(1, 2, 30, 40),
        confidence=0.75,
        class_id=4,
        label="bus",
    )

    converted = to_tracking_detections([source])

    assert converted == (
        TrackingDetection(
            bounding_box=BoundingBox(1.0, 2.0, 30.0, 40.0),
            confidence=0.75,
            class_id=4,
            label="bus",
        ),
    )
    assert isinstance(converted, tuple)


class FakeDetectionModel:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, frame: np.ndarray) -> DetectionResult:
        self.predict_calls += 1
        assert frame.shape == (8, 12, 3)
        return DetectionResult(
            detections=[
                Detection(
                    xyxy=(1, 2, 7, 6),
                    confidence=0.65,
                    class_id=3,
                    label="bike",
                )
            ],
            names={3: "bike"},
        )


def test_object_detection_tracking_command_accepts_current_detection_model() -> None:
    detection_model = FakeDetectionModel()
    command = RunObjectDetectionTrackingCommand(
        detection_model=detection_model,
        algorithm=DetectionAsTrackAlgorithm(),
    )

    result = command.execute(frame=np.zeros((8, 12, 3), dtype=np.uint8))

    assert detection_model.predict_calls == 1
    assert command.frame_index == 1
    assert len(result.tracks) == 1
    assert result.tracks[0].bounding_box == BoundingBox(1, 2, 7, 6)
    assert result.tracks[0].confidence == 0.65
    assert result.tracks[0].class_id == 3
    assert result.tracks[0].label == "bike"


def test_object_detection_tracking_command_reset_resets_tracking_session() -> None:
    command = RunObjectDetectionTrackingCommand(
        detection_model=FakeDetectionModel(),
        algorithm=DetectionAsTrackAlgorithm(),
    )
    first = command.execute(frame=np.zeros((8, 12, 3), dtype=np.uint8))

    command.reset()
    restarted = command.execute(frame=np.zeros((8, 12, 3), dtype=np.uint8))

    assert first.tracks[0].track_id == 1
    assert restarted.tracks[0].track_id == 1
    assert command.frame_index == 1


def test_object_detection_tracking_command_does_not_retain_frame() -> None:
    command = RunObjectDetectionTrackingCommand(
        detection_model=FakeDetectionModel(),
        algorithm=DetectionAsTrackAlgorithm(),
    )
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    frame_reference = weakref.ref(frame)

    command.execute(frame=frame)
    del frame
    gc.collect()

    assert frame_reference() is None
