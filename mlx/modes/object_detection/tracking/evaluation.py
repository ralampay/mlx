from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking.mot import MOTRecord, read_mot_file


def load_mot_ground_truth(path: Path) -> tuple[MOTRecord, ...]:
    records = read_mot_file(path, ground_truth=True)
    if not records:
        raise MLXUserError(f"Ground-truth MOT file contains no evaluable rows: {path}")
    return records


def require_motmetrics():
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise MLXUserError(
            "MOT benchmarking requires motmetrics. Install MLX with the "
            "'tracking' optional dependency and try again."
        ) from exc
    return mm


@dataclass(frozen=True, slots=True)
class TrackingBenchmarkResult:
    frames: int
    ground_truth_objects: int
    predictions: int
    matches: int
    false_positives: int
    misses: int
    id_switches: int
    mota: float
    motp: float
    idf1: float
    precision: float
    recall: float

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            name: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for name, value in asdict(self).items()
        }


class BenchmarkMOTTracking:
    def __init__(
        self,
        *,
        ground_truth_path: Path,
        predictions_path: Path,
        output_path: Path,
        iou_threshold: float = 0.5,
        processed_frames: int | None = None,
        overwrite: bool = False,
    ) -> None:
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path
        self.output_path = output_path
        self.iou_threshold = iou_threshold
        self.processed_frames = processed_frames
        self.overwrite = overwrite

    def execute(self) -> TrackingBenchmarkResult:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise MLXUserError("--benchmark-iou must be greater than 0 and at most 1.")
        if self.output_path.exists() and not self.overwrite:
            raise MLXUserError(
                f"Tracking metrics already exist: {self.output_path}. "
                "Pass --overwrite to replace them."
            )
        ground_truth = load_mot_ground_truth(self.ground_truth_path)
        predictions = read_mot_file(self.predictions_path)
        max_ground_truth_frame = max(
            (record.frame_id for record in ground_truth),
            default=0,
        )
        if self.processed_frames is not None and max_ground_truth_frame > self.processed_frames:
            raise MLXUserError(
                f"Ground truth contains frame {max_ground_truth_frame}, but the video "
                f"produced only {self.processed_frames} frame(s). Check that the files match."
            )
        max_prediction_frame = max(
            (record.frame_id for record in predictions),
            default=0,
        )
        if self.processed_frames is not None and max_prediction_frame > self.processed_frames:
            raise MLXUserError(
                f"Predictions contain frame {max_prediction_frame}, but the video produced "
                f"only {self.processed_frames} frame(s)."
            )
        result = self._evaluate(ground_truth, predictions)
        self._write_result(result)
        return result

    def _evaluate(
        self,
        ground_truth: tuple[MOTRecord, ...],
        predictions: tuple[MOTRecord, ...],
    ) -> TrackingBenchmarkResult:
        mm = require_motmetrics()

        ground_truth_by_frame = _group_by_frame(ground_truth)
        predictions_by_frame = _group_by_frame(predictions)
        max_frame = max(
            max(ground_truth_by_frame, default=0),
            max(predictions_by_frame, default=0),
            self.processed_frames or 0,
        )
        accumulator = mm.MOTAccumulator(auto_id=False)
        for frame_id in range(1, max_frame + 1):
            frame_ground_truth = ground_truth_by_frame.get(frame_id, ())
            frame_predictions = predictions_by_frame.get(frame_id, ())
            distances = _iou_distance_matrix(
                frame_ground_truth,
                frame_predictions,
                threshold=self.iou_threshold,
            )
            accumulator.update(
                [record.track_id for record in frame_ground_truth],
                [record.track_id for record in frame_predictions],
                distances,
                frameid=frame_id,
            )

        metric_names = (
            "num_frames",
            "num_objects",
            "num_matches",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "mota",
            "motp",
            "idf1",
            "precision",
            "recall",
        )
        summary = mm.metrics.create().compute(
            accumulator,
            metrics=list(metric_names),
            name="tracking",
        ).loc["tracking"]
        raw_motp = float(summary["motp"])
        motp = 1.0 - raw_motp if math.isfinite(raw_motp) else math.nan
        return TrackingBenchmarkResult(
            frames=int(summary["num_frames"]),
            ground_truth_objects=int(summary["num_objects"]),
            predictions=len(predictions),
            matches=int(summary["num_matches"]),
            false_positives=int(summary["num_false_positives"]),
            misses=int(summary["num_misses"]),
            id_switches=int(summary["num_switches"]),
            mota=float(summary["mota"]),
            motp=motp,
            idf1=float(summary["idf1"]),
            precision=float(summary["precision"]),
            recall=float(summary["recall"]),
        )

    def _write_result(self, result: TrackingBenchmarkResult) -> None:
        temporary_path = self.output_path.with_name(f".{self.output_path.name}.tmp")
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(
                json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.output_path)
        except OSError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                raise MLXUserError(
                    f"Unable to write tracking metrics: {exc}. Cleanup of temporary "
                    f"file '{temporary_path}' also failed: {cleanup_exc}"
                ) from exc
            raise MLXUserError(f"Unable to write tracking metrics: {exc}") from exc


def _group_by_frame(records: tuple[MOTRecord, ...]) -> dict[int, tuple[MOTRecord, ...]]:
    grouped: dict[int, list[MOTRecord]] = {}
    for record in records:
        grouped.setdefault(record.frame_id, []).append(record)
    return {frame_id: tuple(rows) for frame_id, rows in grouped.items()}


def _iou_distance_matrix(
    ground_truth: tuple[MOTRecord, ...],
    predictions: tuple[MOTRecord, ...],
    *,
    threshold: float,
) -> np.ndarray:
    distances = np.full((len(ground_truth), len(predictions)), np.nan, dtype=np.float64)
    for ground_truth_index, expected in enumerate(ground_truth):
        for prediction_index, actual in enumerate(predictions):
            iou = _record_iou(expected, actual)
            if iou >= threshold:
                distances[ground_truth_index, prediction_index] = 1.0 - iou
    return distances


def _record_iou(first: MOTRecord, second: MOTRecord) -> float:
    right = min(first.left + first.width, second.left + second.width)
    bottom = min(first.top + first.height, second.top + second.height)
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0.0 else 0.0
