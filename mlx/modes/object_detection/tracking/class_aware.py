from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking.models import (
    BoundingBox,
    TrackStatus,
    TrackingFrameResult,
)
from mlx.modes.object_detection.tracking.mot import MOTRecord, MOTResultWriter


CLASS_AWARE_SCHEMA_VERSION = "mlx.tracking.record/v1"


@dataclass(frozen=True, slots=True)
class ClassAwareTrackingRecord:
    frame_id: int
    track_id: int
    class_id: int
    label: str | None
    bounding_box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        for field_name in ("frame_id", "track_id", "class_id"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Tracking record {field_name} must be an integer."
                )
        if self.frame_id < 1:
            raise ValueError("Tracking record frame IDs must be one or greater.")
        if self.track_id < 1:
            raise ValueError("Tracking record track IDs must be one or greater.")
        if self.class_id < 0:
            raise ValueError("Tracking record class IDs must be zero or greater.")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("Tracking record labels must be strings or null.")
        if isinstance(self.confidence, bool):
            raise ValueError("Tracking record confidence must be a finite number.")
        confidence = float(self.confidence)
        if not math.isfinite(confidence):
            raise ValueError("Tracking record confidence must be a finite number.")
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CLASS_AWARE_SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "label": self.label,
            "bounding_box": {
                "x1": self.bounding_box.x1,
                "y1": self.bounding_box.y1,
                "x2": self.bounding_box.x2,
                "y2": self.bounding_box.y2,
            },
            "confidence": self.confidence,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)

    def to_mot_record(self) -> MOTRecord:
        return MOTRecord(
            frame_id=self.frame_id,
            track_id=self.track_id,
            left=self.bounding_box.x1,
            top=self.bounding_box.y1,
            width=self.bounding_box.width,
            height=self.bounding_box.height,
            confidence=self.confidence,
        )


def tracking_result_to_class_aware_records(
    result: TrackingFrameResult,
) -> tuple[ClassAwareTrackingRecord, ...]:
    records = []
    for track in sorted(result.tracks, key=lambda item: item.track_id):
        if track.status is not TrackStatus.CONFIRMED:
            continue
        if track.last_seen_frame != result.frame_index:
            continue
        records.append(
            ClassAwareTrackingRecord(
                frame_id=result.frame_index,
                track_id=track.track_id,
                class_id=track.class_id,
                label=track.label,
                bounding_box=track.bounding_box,
                confidence=track.confidence,
            )
        )
    return tuple(records)


class ClassAwareTrackingResultWriter:
    """Stream current confirmed tracks to an atomic JSON Lines artifact."""

    def __init__(self, *, output_dir: Path, overwrite: bool = False) -> None:
        self.output_dir = output_dir
        self.output_path = output_dir / "tracks.jsonl"
        self._temporary_path = output_dir / ".tracks.jsonl.tmp"
        self.overwrite = overwrite
        self._stream = None
        self.rows_written = 0

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError(
                "ClassAwareTrackingResultWriter has already been started."
            )
        if self.output_path.exists() and not self.overwrite:
            raise MLXUserError(
                f"Class-aware tracking result already exists: {self.output_path}. "
                "Pass --overwrite to replace it."
            )
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stream = self._temporary_path.open("w", encoding="utf-8")
        except OSError as exc:
            raise MLXUserError(
                f"Unable to create class-aware tracking output under "
                f"'{self.output_dir}': {exc}"
            ) from exc

    def write(self, result: TrackingFrameResult) -> int:
        if self._stream is None:
            raise RuntimeError(
                "ClassAwareTrackingResultWriter.start() must be called before write()."
            )
        records = tracking_result_to_class_aware_records(result)
        try:
            for record in records:
                self._stream.write(record.to_json_line() + "\n")
        except OSError as exc:
            raise MLXUserError(
                f"Unable to write class-aware tracking results: {exc}"
            ) from exc
        self.rows_written += len(records)
        return len(records)

    def finalize(self) -> Path:
        if self._stream is None:
            raise RuntimeError(
                "ClassAwareTrackingResultWriter.start() must be called before finalize()."
            )
        try:
            self._stream.close()
            self._stream = None
            self._temporary_path.replace(self.output_path)
        except OSError as exc:
            raise MLXUserError(
                f"Unable to finalize class-aware tracking results: {exc}"
            ) from exc
        return self.output_path

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        try:
            self._temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise MLXUserError(
                f"Unable to remove incomplete class-aware tracking output "
                f"'{self._temporary_path}': {exc}"
            ) from exc


class TrackingResultWriter:
    """Compose canonical MOT and lossless class-aware tracking writers."""

    def __init__(
        self,
        *,
        output_dir: Path,
        overwrite: bool = False,
        mot_writer: MOTResultWriter | None = None,
        class_aware_writer: ClassAwareTrackingResultWriter | None = None,
    ) -> None:
        self.mot_writer = mot_writer or MOTResultWriter(
            output_dir=output_dir,
            overwrite=overwrite,
        )
        self.class_aware_writer = class_aware_writer or (
            ClassAwareTrackingResultWriter(
                output_dir=output_dir,
                overwrite=overwrite,
            )
        )

    @property
    def output_path(self) -> Path:
        return self.mot_writer.output_path

    @property
    def class_aware_output_path(self) -> Path:
        return self.class_aware_writer.output_path

    @property
    def rows_written(self) -> int:
        return self.mot_writer.rows_written

    def start(self) -> None:
        self.mot_writer.start()
        try:
            self.class_aware_writer.start()
        except Exception as exc:
            try:
                self.mot_writer.abort()
            except Exception as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"Cleanup of the incomplete MOT output also failed: "
                        f"{cleanup_exc}"
                    )
            raise

    def write(self, result: TrackingFrameResult) -> int:
        mot_count = self.mot_writer.write(result)
        class_aware_count = self.class_aware_writer.write(result)
        if mot_count != class_aware_count:
            raise RuntimeError(
                "MOT and class-aware writers produced different tracking row counts."
            )
        return mot_count

    def finalize(self) -> Path:
        class_aware_path = self.class_aware_writer.finalize()
        try:
            return self.mot_writer.finalize()
        except Exception as exc:
            try:
                class_aware_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "Cleanup of the finalized class-aware output also failed: "
                        f"{cleanup_exc}"
                    )
            raise

    def abort(self) -> None:
        failures = []
        for writer in (self.class_aware_writer, self.mot_writer):
            try:
                writer.abort()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise MLXUserError(
                "Unable to remove incomplete tracking outputs: "
                + "; ".join(str(failure) for failure in failures)
            ) from failures[0]


def read_class_aware_tracking_file(
    path: Path,
) -> tuple[ClassAwareTrackingRecord, ...]:
    if not path.is_file():
        raise MLXUserError(f"Class-aware tracking JSONL file not found: {path}")
    records = []
    seen_track_frames: set[tuple[int, int]] = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = _parse_class_aware_record(
                    line,
                    path=path,
                    line_number=line_number,
                )
                track_frame = (record.frame_id, record.track_id)
                if track_frame in seen_track_frames:
                    raise MLXUserError(
                        f"Invalid class-aware tracking file '{path}': track "
                        f"{record.track_id} appears more than once in frame "
                        f"{record.frame_id}."
                    )
                seen_track_frames.add(track_frame)
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise MLXUserError(
            f"Unable to read class-aware tracking file '{path}': {exc}"
        ) from exc
    return tuple(records)


@dataclass(frozen=True, slots=True)
class MOTExportResult:
    output_path: Path
    rows_written: int
    source_rows: int
    class_ids: tuple[int, ...]


class ExportMOTFromClassAwareTracking:
    """Validate class-aware JSONL and export standard 10-column MOT predictions."""

    def __init__(
        self,
        *,
        source_path: Path,
        output_dir: Path,
        class_ids: tuple[int, ...] = (),
        overwrite: bool = False,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.source_path = source_path
        self.output_dir = output_dir
        self.class_ids = tuple(class_ids)
        self.overwrite = overwrite
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> MOTExportResult:
        if any(class_id < 0 for class_id in self.class_ids):
            raise MLXUserError("MOT export class IDs must be zero or greater.")
        records = read_class_aware_tracking_file(self.source_path)
        selected_class_ids = set(self.class_ids)
        selected = tuple(
            record
            for record in records
            if not selected_class_ids or record.class_id in selected_class_ids
        )
        output_path = self.output_dir / "tracks.txt"
        if output_path.exists() and not self.overwrite:
            raise MLXUserError(
                f"MOT tracking result already exists: {output_path}. "
                "Pass --overwrite to replace it."
            )
        temporary_path = self.output_dir / ".tracks.txt.tmp"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(
                "".join(
                    record.to_mot_record().to_line() + "\n"
                    for record in sorted(
                        selected,
                        key=lambda item: (item.frame_id, item.track_id),
                    )
                ),
                encoding="utf-8",
            )
            temporary_path.replace(output_path)
        except OSError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"Cleanup of temporary MOT output also failed: {cleanup_exc}"
                    )
            raise MLXUserError(
                f"Unable to export MOT tracking results under '{self.output_dir}': "
                f"{exc}"
            ) from exc
        emit(
            self.reporter,
            "success",
            f"Exported {len(selected)} MOT row(s) from {len(records)} class-aware "
            f"tracking record(s) to {output_path}.",
        )
        return MOTExportResult(
            output_path=output_path,
            rows_written=len(selected),
            source_rows=len(records),
            class_ids=self.class_ids,
        )


def _parse_class_aware_record(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> ClassAwareTrackingRecord:
    try:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("row must be a JSON object")
        schema_version = value["schema_version"]
        if schema_version != CLASS_AWARE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {schema_version!r}; expected "
                f"{CLASS_AWARE_SCHEMA_VERSION!r}"
            )
        box = value["bounding_box"]
        if not isinstance(box, dict):
            raise ValueError("bounding_box must be a JSON object")
        return ClassAwareTrackingRecord(
            frame_id=_require_json_integer(value["frame_id"], "frame_id"),
            track_id=_require_json_integer(value["track_id"], "track_id"),
            class_id=_require_json_integer(value["class_id"], "class_id"),
            label=value.get("label"),
            bounding_box=BoundingBox(
                x1=_require_json_number(box["x1"], "bounding_box.x1"),
                y1=_require_json_number(box["y1"], "bounding_box.y1"),
                x2=_require_json_number(box["x2"], "bounding_box.x2"),
                y2=_require_json_number(box["y2"], "bounding_box.y2"),
            ),
            confidence=_require_json_number(value["confidence"], "confidence"),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MLXUserError(
            f"Invalid class-aware tracking row at {path}:{line_number}: {exc}"
        ) from exc


def _require_json_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a JSON integer")
    return value


def _require_json_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number")
    return float(value)
