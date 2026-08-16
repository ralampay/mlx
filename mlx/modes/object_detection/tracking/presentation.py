from __future__ import annotations

import hashlib

import cv2
import numpy as np
from rich.table import Table

from mlx.core.ui import console
from mlx.modes.object_detection.tracking.evaluation import TrackingBenchmarkResult
from mlx.modes.object_detection.tracking.models import TrackingFrameResult


def annotate_tracks(
    frame: np.ndarray,
    result: TrackingFrameResult,
) -> np.ndarray:
    """Render current track observations without mutating the source frame."""

    annotated = frame.copy()
    visible_tracks = tuple(
        track
        for track in result.tracks
        if track.last_seen_frame == result.frame_index
    )
    for track in visible_tracks:
        x1, y1, x2, y2 = (
            int(round(value)) for value in track.bounding_box.as_xyxy()
        )
        color = _color_for_track(track.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = track.label or f"class {track.class_id}"
        text = (
            f"ID {track.track_id} | {label} | {track.confidence:.2f} | "
            f"{track.status.value}"
        )
        _draw_label(annotated, text=text, origin=(x1, y1), color=color)

    summary = f"Frame {result.frame_index} | visible tracks {len(visible_tracks)}"
    _draw_label(annotated, text=summary, origin=(8, 24), color=(255, 255, 255))
    return annotated


def _draw_label(
    frame: np.ndarray,
    *,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    text_y = max(y - 6, text_height + baseline + 2)
    top = max(text_y - text_height - 3, 0)
    right = min(x + text_width + 6, frame.shape[1] - 1)
    cv2.rectangle(frame, (max(x, 0), top), (right, text_y + baseline + 2), color, -1)
    text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
    cv2.putText(
        frame,
        text,
        (max(x + 3, 0), text_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(track_id).encode("utf-8")).digest()
    return tuple(max(channel, 64) for channel in digest[:3])


def print_trackers(trackers: tuple[str, ...]) -> None:
    table = Table(title="Tracking Algorithms")
    table.add_column("Name", style="cyan")
    for tracker in trackers:
        table.add_row(tracker)
    console.print(table)


def print_tracking_benchmark(result: TrackingBenchmarkResult) -> None:
    table = Table(title="MOT Tracking Benchmark", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="white")
    rows = (
        ("Frames", str(result.frames)),
        ("Ground-truth objects", str(result.ground_truth_objects)),
        ("Predictions", str(result.predictions)),
        ("Matches", str(result.matches)),
        ("False positives", str(result.false_positives)),
        ("Misses", str(result.misses)),
        ("ID switches", str(result.id_switches)),
        ("MOTA", _format_metric(result.mota)),
        ("MOTP (mean matched IoU)", _format_metric(result.motp)),
        ("IDF1", _format_metric(result.idf1)),
        ("Precision", _format_metric(result.precision)),
        ("Recall", _format_metric(result.recall)),
    )
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"
