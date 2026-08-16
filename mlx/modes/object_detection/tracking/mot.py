from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking.models import (
    TrackStatus,
    TrackingFrameResult,
)


@dataclass(frozen=True, slots=True)
class MOTRecord:
    frame_id: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float
    world_x: float = -1.0
    world_y: float = -1.0
    world_z: float = -1.0

    def __post_init__(self) -> None:
        if self.frame_id < 1:
            raise ValueError("MOT frame IDs must be one or greater.")
        if self.track_id < 1:
            raise ValueError("MOT track IDs must be one or greater.")
        for name in (
            "left",
            "top",
            "width",
            "height",
            "confidence",
            "world_x",
            "world_y",
            "world_z",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"MOT {name} must be a finite number.")
        if self.width < 0.0 or self.height < 0.0:
            raise ValueError("MOT bounding-box width and height cannot be negative.")

    def to_line(self) -> str:
        values = (
            str(self.frame_id),
            str(self.track_id),
            _format_float(self.left),
            _format_float(self.top),
            _format_float(self.width),
            _format_float(self.height),
            _format_float(self.confidence),
            _format_float(self.world_x),
            _format_float(self.world_y),
            _format_float(self.world_z),
        )
        return ",".join(values)


def tracking_result_to_mot_records(
    result: TrackingFrameResult,
) -> tuple[MOTRecord, ...]:
    records = []
    for track in sorted(result.tracks, key=lambda item: item.track_id):
        if track.status is not TrackStatus.CONFIRMED:
            continue
        if track.last_seen_frame != result.frame_index:
            continue
        box = track.bounding_box
        records.append(
            MOTRecord(
                frame_id=result.frame_index,
                track_id=track.track_id,
                left=box.x1,
                top=box.y1,
                width=box.width,
                height=box.height,
                confidence=track.confidence,
            )
        )
    return tuple(records)


class MOTResultWriter:
    """Stream tracking results to a temporary MOT file, then finalize atomically."""

    def __init__(self, *, output_dir: Path, overwrite: bool = False) -> None:
        self.output_dir = output_dir
        self.output_path = output_dir / "tracks.txt"
        self._temporary_path = output_dir / ".tracks.txt.tmp"
        self.overwrite = overwrite
        self._stream = None
        self.rows_written = 0

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("MOTResultWriter has already been started.")
        if self.output_path.exists() and not self.overwrite:
            raise MLXUserError(
                f"Tracking result already exists: {self.output_path}. "
                "Pass --overwrite to replace it."
            )
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stream = self._temporary_path.open("w", encoding="utf-8")
        except OSError as exc:
            raise MLXUserError(
                f"Unable to create tracking output under '{self.output_dir}': {exc}"
            ) from exc

    def write(self, result: TrackingFrameResult) -> int:
        if self._stream is None:
            raise RuntimeError("MOTResultWriter.start() must be called before write().")
        records = tracking_result_to_mot_records(result)
        try:
            for record in records:
                self._stream.write(record.to_line() + "\n")
        except OSError as exc:
            raise MLXUserError(f"Unable to write tracking results: {exc}") from exc
        self.rows_written += len(records)
        return len(records)

    def finalize(self) -> Path:
        if self._stream is None:
            raise RuntimeError("MOTResultWriter.start() must be called before finalize().")
        try:
            self._stream.close()
            self._stream = None
            self._temporary_path.replace(self.output_path)
        except OSError as exc:
            raise MLXUserError(f"Unable to finalize tracking results: {exc}") from exc
        return self.output_path

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        try:
            self._temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise MLXUserError(
                f"Unable to remove incomplete tracking output '{self._temporary_path}': {exc}"
            ) from exc


def read_mot_file(path: Path, *, ground_truth: bool = False) -> tuple[MOTRecord, ...]:
    if not path.is_file():
        label = "Ground-truth" if ground_truth else "Tracking result"
        raise MLXUserError(f"{label} MOT file not found: {path}")
    records = []
    seen_track_frames: set[tuple[int, int]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MLXUserError(f"Unable to read MOT file '{path}': {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 10:
            raise MLXUserError(
                f"Invalid MOT row at {path}:{line_number}: expected 10 comma-separated "
                f"fields, found {len(values)}."
            )
        try:
            record = MOTRecord(
                frame_id=int(values[0]),
                track_id=int(values[1]),
                left=float(values[2]),
                top=float(values[3]),
                width=float(values[4]),
                height=float(values[5]),
                confidence=float(values[6]),
                world_x=float(values[7]),
                world_y=float(values[8]),
                world_z=float(values[9]),
            )
        except ValueError as exc:
            raise MLXUserError(f"Invalid MOT row at {path}:{line_number}: {exc}") from exc
        if ground_truth and record.confidence <= 0.0:
            continue
        track_frame = (record.frame_id, record.track_id)
        if track_frame in seen_track_frames:
            raise MLXUserError(
                f"Invalid MOT file '{path}': track {record.track_id} appears more than "
                f"once in frame {record.frame_id}."
            )
        seen_track_frames.add(track_frame)
        records.append(record)
    return tuple(records)


def _format_float(value: float) -> str:
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted if formatted not in {"", "-0"} else "0"
