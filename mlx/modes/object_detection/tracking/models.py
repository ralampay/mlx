from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


def _finite_float(value: float, *, field_name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be a finite number.")
    return converted


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An immutable axis-aligned bounding box in ``(x1, y1, x2, y2)`` form."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for field_name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name=field_name),
            )
        if self.x2 < self.x1:
            raise ValueError("Bounding-box x2 must be greater than or equal to x1.")
        if self.y2 < self.y1:
            raise ValueError("Bounding-box y2 must be greater than or equal to y1.")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class TrackingDetection:
    """A detector-independent detection supplied to a tracking algorithm."""

    bounding_box: BoundingBox
    confidence: float
    class_id: int
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence",
            _finite_float(self.confidence, field_name="confidence"),
        )


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass(slots=True)
class TrackState:
    """Mutable internal state for future online tracking implementations."""

    track_id: int
    bounding_box: BoundingBox
    confidence: float
    class_id: int
    label: str | None
    created_at_frame: int
    last_seen_frame: int
    hits: int = 1
    missing_frames: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE

    def __post_init__(self) -> None:
        self.confidence = _finite_float(self.confidence, field_name="confidence")


@dataclass(frozen=True, slots=True)
class TrackResult:
    """Immutable public snapshot of one track at a particular frame."""

    track_id: int
    bounding_box: BoundingBox
    confidence: float
    class_id: int
    label: str | None
    status: TrackStatus
    hits: int
    missing_frames: int
    last_seen_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence",
            _finite_float(self.confidence, field_name="confidence"),
        )


@dataclass(frozen=True, slots=True)
class TrackingFrameResult:
    """Immutable tracking output for one frame; the source frame is not retained."""

    frame_index: int
    tracks: tuple[TrackResult, ...]

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("Tracking frame index must be zero or greater.")
        object.__setattr__(self, "tracks", tuple(self.tracks))
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Tracking frame result contains duplicate track IDs.")


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Association indices where each match is ``(track_id, detection_index)``."""

    matches: tuple[tuple[int, int], ...]
    unmatched_track_ids: tuple[int, ...]
    unmatched_detection_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        raw_matches = tuple(tuple(match) for match in self.matches)
        if any(len(match) != 2 for match in raw_matches):
            raise ValueError(
                "Each association match must contain a track ID and detection index."
            )
        matches = tuple((match[0], match[1]) for match in raw_matches)
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "unmatched_track_ids", tuple(self.unmatched_track_ids))
        object.__setattr__(
            self,
            "unmatched_detection_indices",
            tuple(self.unmatched_detection_indices),
        )
