from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking.class_aware import (
    ClassAwareTrackingRecord,
    read_class_aware_tracking_file,
)
from mlx.modes.object_detection.tracking.mot import MOTRecord, read_mot_file
from mlx.modes.object_detection.tracking.replay_html import (
    render_tracking_replay_html,
)


REPLAY_SCHEMA_VERSION = "mlx.tracking.replay/v1"


@dataclass(frozen=True, slots=True)
class TrackingReplayResult:
    data_path: Path
    html_path: Path
    frame_count: int
    prediction_track_count: int
    ground_truth_track_count: int


class ExportTrackingReplay:
    """Export MOT results as portable data and a self-contained 2D HTML replay."""

    def __init__(
        self,
        *,
        predictions_path: Path,
        output_dir: Path,
        class_aware_path: Path | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        frame_count: int | None = None,
        fps: float = 30.0,
        ground_truth_path: Path | None = None,
        metrics: Mapping[str, int | float | None] | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        source_name: str | None = None,
        overwrite: bool = False,
        html_renderer: Callable[[Mapping[str, Any]], str] = (
            render_tracking_replay_html
        ),
    ) -> None:
        self.predictions_path = predictions_path
        self.output_dir = output_dir
        self.class_aware_path = class_aware_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_count = frame_count
        self.fps = fps
        self.ground_truth_path = ground_truth_path
        self.metrics = metrics
        self.run_metadata = run_metadata or {}
        self.source_name = source_name
        self.overwrite = overwrite
        self.html_renderer = html_renderer

    def execute(self) -> TrackingReplayResult:
        self._validate_options()
        data_path = self.output_dir / "replay.json"
        html_path = self.output_dir / "replay.html"
        self._validate_outputs(data_path, html_path)

        predictions = read_mot_file(self.predictions_path)
        class_aware_records = (
            read_class_aware_tracking_file(self.class_aware_path)
            if self.class_aware_path is not None
            else ()
        )
        class_metadata = _validate_class_metadata(
            predictions,
            class_aware_records,
            class_aware_path=self.class_aware_path,
        )
        ground_truth = (
            read_mot_file(self.ground_truth_path, ground_truth=True)
            if self.ground_truth_path is not None
            else ()
        )
        if self.frame_count is not None:
            prediction_frame = max(
                (record.frame_id for record in predictions),
                default=0,
            )
            if prediction_frame > self.frame_count:
                raise MLXUserError(
                    f"Predictions contain frame {prediction_frame}, beyond replay "
                    f"frame count {self.frame_count}."
                )
            ground_truth = tuple(
                record
                for record in ground_truth
                if record.frame_id <= self.frame_count
            )
        payload = self._build_payload(
            predictions,
            ground_truth,
            class_metadata=class_metadata,
        )
        try:
            data_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
            html_text = self.html_renderer(payload)
        except (TypeError, ValueError) as exc:
            raise MLXUserError(
                f"Unable to serialize tracking replay data: {exc}. "
                "Check the supplied run metadata and metrics."
            ) from exc
        self._write_outputs(
            data_path=data_path,
            data_text=data_text,
            html_path=html_path,
            html_text=html_text,
        )
        return TrackingReplayResult(
            data_path=data_path,
            html_path=html_path,
            frame_count=payload["frame_count"],
            prediction_track_count=payload["predictions"]["track_count"],
            ground_truth_track_count=(
                payload["ground_truth"]["track_count"]
                if payload["ground_truth"] is not None
                else 0
            ),
        )

    def _validate_options(self) -> None:
        if self.frame_width is not None and self.frame_width < 1:
            raise MLXUserError("Tracking replay frame width must be one or greater.")
        if self.frame_height is not None and self.frame_height < 1:
            raise MLXUserError("Tracking replay frame height must be one or greater.")
        if self.frame_count is not None and self.frame_count < 1:
            raise MLXUserError("Tracking replay frame count must be one or greater.")
        if not math.isfinite(float(self.fps)) or self.fps <= 0:
            raise MLXUserError("Tracking replay FPS must be a finite value above zero.")

    def _validate_outputs(self, data_path: Path, html_path: Path) -> None:
        if self.overwrite:
            return
        existing = next((path for path in (data_path, html_path) if path.exists()), None)
        if existing is not None:
            raise MLXUserError(
                f"Tracking replay artifact already exists: {existing}. "
                "Pass --overwrite to replace it."
            )

    def _build_payload(
        self,
        predictions: tuple[MOTRecord, ...],
        ground_truth: tuple[MOTRecord, ...],
        *,
        class_metadata: Mapping[tuple[int, int], ClassAwareTrackingRecord],
    ) -> dict[str, Any]:
        all_records = predictions + ground_truth
        max_frame = max((record.frame_id for record in all_records), default=0)
        frame_count = max(self.frame_count or 0, max_frame)
        if frame_count < 1:
            raise MLXUserError(
                "Cannot create a tracking replay with no frames. Provide a positive "
                "frame count or a non-empty MOT file."
            )
        frame_width, frame_height = self._resolve_canvas(all_records)
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "coordinate_system": {
                "origin": "top-left",
                "units": "pixels",
                "bounding_box": "left,top,width,height",
                "y_axis": "down",
            },
            "source": {"name": self.source_name},
            "canvas": {"width": frame_width, "height": frame_height},
            "frame_count": frame_count,
            "fps": float(self.fps),
            "run": dict(self.run_metadata),
            "predictions": _record_collection(
                predictions,
                class_metadata=class_metadata,
            ),
            "ground_truth": (
                _record_collection(ground_truth) if self.ground_truth_path else None
            ),
            "metrics": dict(self.metrics) if self.metrics is not None else None,
        }

    def _resolve_canvas(
        self,
        records: tuple[MOTRecord, ...],
    ) -> tuple[int, int]:
        inferred_width = math.ceil(
            max((record.left + record.width for record in records), default=0.0)
        )
        inferred_height = math.ceil(
            max((record.top + record.height for record in records), default=0.0)
        )
        width = self.frame_width if self.frame_width is not None else inferred_width
        height = self.frame_height if self.frame_height is not None else inferred_height
        if width < 1 or height < 1:
            raise MLXUserError(
                "Cannot infer replay canvas dimensions from empty tracking results. "
                "Provide frame_width and frame_height."
            )
        return width, height

    def _write_outputs(
        self,
        *,
        data_path: Path,
        data_text: str,
        html_path: Path,
        html_text: str,
    ) -> None:
        data_temporary = data_path.with_name(f".{data_path.name}.tmp")
        html_temporary = html_path.with_name(f".{html_path.name}.tmp")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            data_temporary.write_text(data_text, encoding="utf-8")
            html_temporary.write_text(html_text, encoding="utf-8")
            data_temporary.replace(data_path)
            html_temporary.replace(html_path)
        except OSError as exc:
            cleanup_failures = []
            for temporary_path in (data_temporary, html_temporary):
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    cleanup_failures.append(str(cleanup_exc))
            suffix = (
                f" Temporary-file cleanup also failed: {'; '.join(cleanup_failures)}"
                if cleanup_failures
                else ""
            )
            raise MLXUserError(
                f"Unable to write tracking replay under '{self.output_dir}': {exc}."
                f"{suffix}"
            ) from exc


def _record_collection(
    records: tuple[MOTRecord, ...],
    *,
    class_metadata: Mapping[tuple[int, int], ClassAwareTrackingRecord] | None = None,
) -> dict[str, Any]:
    class_metadata = class_metadata or {}
    return {
        "record_count": len(records),
        "track_count": len({record.track_id for record in records}),
        "records": [
            {
                "frame_id": record.frame_id,
                "track_id": record.track_id,
                "left": record.left,
                "top": record.top,
                "width": record.width,
                "height": record.height,
                "confidence": record.confidence,
                "class_id": (
                    class_metadata[(record.frame_id, record.track_id)].class_id
                    if (record.frame_id, record.track_id) in class_metadata
                    else None
                ),
                "label": (
                    class_metadata[(record.frame_id, record.track_id)].label
                    if (record.frame_id, record.track_id) in class_metadata
                    else None
                ),
            }
            for record in records
        ],
    }


def _validate_class_metadata(
    predictions: tuple[MOTRecord, ...],
    class_aware_records: tuple[ClassAwareTrackingRecord, ...],
    *,
    class_aware_path: Path | None,
) -> dict[tuple[int, int], ClassAwareTrackingRecord]:
    if class_aware_path is None:
        return {}
    prediction_by_key = {
        (record.frame_id, record.track_id): record for record in predictions
    }
    class_by_key = {
        (record.frame_id, record.track_id): record
        for record in class_aware_records
    }
    if prediction_by_key.keys() != class_by_key.keys():
        missing = sorted(prediction_by_key.keys() - class_by_key.keys())
        extra = sorted(class_by_key.keys() - prediction_by_key.keys())
        raise MLXUserError(
            f"Class-aware tracking file '{class_aware_path}' does not describe the "
            f"same frame/track rows as the MOT predictions. Missing keys: "
            f"{missing[:3]}; extra keys: {extra[:3]}."
        )
    for key, mot_record in prediction_by_key.items():
        extracted = class_by_key[key].to_mot_record()
        values_match = all(
            math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-6)
            for first, second in (
                (mot_record.left, extracted.left),
                (mot_record.top, extracted.top),
                (mot_record.width, extracted.width),
                (mot_record.height, extracted.height),
                (mot_record.confidence, extracted.confidence),
            )
        )
        if not values_match:
            raise MLXUserError(
                f"Class-aware tracking row {key} does not match its MOT prediction."
            )
    return class_by_key
