from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.presentation import RichWorkflowReporter
from mlx.modes.object_detection.streaming import OpenCVFrameSink
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
    if action == "ls-trackers":
        trackers = list_trackers()
        print_trackers(trackers)
        return trackers
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
            reporter=RichWorkflowReporter(),
        ).execute()
        if result.benchmark is not None:
            print_tracking_benchmark(result.benchmark)
        return result
    raise MLXUserError(
        f"Unsupported action '{action}' for track. Available actions: ls-trackers, run."
    )
