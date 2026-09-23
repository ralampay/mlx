from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.commands import NullWorkflowReporter
from mlx.modes.object_detection.presentation import RichWorkflowReporter
from mlx.modes.object_detection.streaming import OpenCVFrameSink
from mlx.modes.object_detection.tracking.benchmark import BenchmarkTrackingDataset
from mlx.modes.object_detection.tracking.class_aware import (
    ExportMOTFromClassAwareTracking,
)
from mlx.modes.object_detection.tracking.presentation import (
    TrackingTrajectoryRenderer,
    annotate_tracks,
    print_trackers,
    print_tracking_benchmark,
    print_tracking_dataset_benchmark,
)
from mlx.modes.object_detection.tracking.preparation import PrepareTrackingBenchmarks
from mlx.modes.object_detection.tracking.registry import list_trackers
from mlx.modes.object_detection.tracking.requests import (
    TrackingBenchmarkRequest,
    TrackingRequest,
)
from mlx.modes.object_detection.tracking.session import RunTrackingVideo


def run_tracking(config: dict[str, Any]):
    action = config.get("action") or "run"
    is_json = config.get("output_format") == "json"
    reporter = NullWorkflowReporter() if is_json else RichWorkflowReporter()
    if action == "build-dataset":
        if not config.get("dataset_path") or not config.get("output_path"):
            raise MLXUserError("Tracking dataset preparation requires --dataset SOURCE and --output DESTINATION.")
        return PrepareTrackingBenchmarks(Path(config["dataset_path"]), Path(config["output_path"]), reporter=reporter).execute()
    if action == "benchmark":
        if not config.get("dataset_path"):
            raise MLXUserError("Tracking benchmarking requires --dataset pointing to prepared sequences.")
        split = config.get("split")
        if "_explicit_options" in config and "split" not in config["_explicit_options"]:
            split = None
        request = TrackingBenchmarkRequest(
            dataset_path=config["dataset_path"],
            tracking=TrackingRequest.from_config({**config, "display": False} if is_json else config),
            split=split,
            real_time_results=config.get("real_time_results", True),
        )
        result = BenchmarkTrackingDataset(
            request, reporter=reporter,
            display_factory=lambda: OpenCVFrameSink(title="MLX Tracking Benchmark", delay_ms=1),
            renderer_factory=TrackingTrajectoryRenderer,
        ).execute()
        if not is_json:
            print_tracking_dataset_benchmark(result.sequences)
        return result
    if action == "ls-trackers":
        trackers = list_trackers()
        if not is_json:
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
        request = TrackingRequest.from_config(
            {**config, "display": False} if is_json else config
        )
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
        if result.benchmark is not None and not is_json:
            print_tracking_benchmark(result.benchmark)
        return result
    raise MLXUserError(
        f"Unsupported action '{action}' for track. Available actions: "
        "benchmark, build-dataset, export-mot, ls-trackers, run."
    )
