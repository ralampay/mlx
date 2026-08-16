from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.presentation import RichWorkflowReporter
from mlx.modes.object_detection.streaming import OpenCVFrameSink
from mlx.modes.object_detection.tracking.class_aware import (
    ExportMOTFromClassAwareTracking,
)
from mlx.modes.object_detection.tracking.presentation import (
    annotate_tracks,
    print_trackers,
    print_tracking_benchmark,
)
from mlx.modes.object_detection.tracking.registry import list_trackers
from mlx.modes.object_detection.tracking.requests import TrackingRequest
from mlx.modes.object_detection.tracking.session import RunTrackingVideo


def run_tracking(config: dict[str, Any]):
    action = config.get("action") or "run"
    reporter = RichWorkflowReporter()
    if action == "ls-trackers":
        trackers = list_trackers()
        print_trackers(trackers)
        return trackers
    if action == "export-mot":
        source_path = config.get("tracking_jsonl")
        output_path = config.get("output_path")
        if not source_path:
            raise MLXUserError(
                "Tracking MOT export requires --tracking-jsonl pointing to "
                "tracks.jsonl."
            )
        if not output_path:
            raise MLXUserError(
                "Tracking MOT export requires --output pointing to a result directory."
            )
        return ExportMOTFromClassAwareTracking(
            source_path=Path(source_path).expanduser(),
            output_dir=Path(output_path).expanduser(),
            class_ids=tuple(config.get("track_class_ids") or ()),
            overwrite=bool(config.get("overwrite", False)),
            reporter=reporter,
        ).execute()
    if action == "run":
        request = TrackingRequest.from_config(config)
        sink = (
            OpenCVFrameSink(title="MLX Tracking", delay_ms=10)
            if request.display
            else None
        )
        result = RunTrackingVideo(
            request,
            frame_sink=sink,
            renderer=annotate_tracks if sink is not None else None,
            reporter=reporter,
        ).execute()
        if result.benchmark is not None:
            print_tracking_benchmark(result.benchmark)
        return result
    raise MLXUserError(
        f"Unsupported action '{action}' for track. Available actions: "
        "export-mot, ls-trackers, run."
    )
