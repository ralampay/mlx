from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.commands import CreateObjectDetector
from mlx.modes.object_detection.models import DetectionAdapter
from mlx.modes.object_detection.requests import ObjectDetectionRequest
from mlx.modes.object_detection.streaming import (
    FrameSink,
    FrameSource,
    FrameSourceMetadata,
    MetadataFrameSource,
    OpenCVFrameSource,
)
from mlx.modes.object_detection.tracking.detection import (
    RunObjectDetectionTrackingCommand,
)
from mlx.modes.object_detection.tracking.evaluation import (
    BenchmarkMOTTracking,
    TrackingBenchmarkResult,
    load_mot_ground_truth,
    require_motmetrics,
)
from mlx.modes.object_detection.tracking.models import TrackingFrameResult
from mlx.modes.object_detection.tracking.mot import MOTResultWriter
from mlx.modes.object_detection.tracking.protocols import TrackingAlgorithm
from mlx.modes.object_detection.tracking.registry import CreateTrackingAlgorithm
from mlx.modes.object_detection.tracking.replay import (
    ExportTrackingReplay,
    TrackingReplayResult,
)
from mlx.modes.object_detection.tracking.requests import TrackingRequest


@dataclass(frozen=True, slots=True)
class TrackingRunResult:
    frames_processed: int
    tracks_written: int
    output_path: Path
    metrics_path: Path | None = None
    benchmark: TrackingBenchmarkResult | None = None
    stopped_by_user: bool = False
    replay_data_path: Path | None = None
    replay_html_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _ProcessedTrackingVideo:
    frames_processed: int
    output_path: Path
    stopped_by_user: bool
    frame_width: int
    frame_height: int


class RunTrackingVideo:
    """Run detection and stateful tracking over a video and persist MOT results."""

    def __init__(
        self,
        request: TrackingRequest,
        *,
        detector: DetectionAdapter | None = None,
        algorithm: TrackingAlgorithm | None = None,
        frame_source: FrameSource | None = None,
        frame_sink: FrameSink | None = None,
        renderer: Callable[[np.ndarray, TrackingFrameResult], np.ndarray] | None = None,
        writer: MOTResultWriter | None = None,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.request = request
        self.detector = detector
        self.algorithm = algorithm
        self.frame_source = frame_source
        if (frame_sink is None) != (renderer is None):
            raise ValueError("Tracking display requires both a frame sink and renderer.")
        self.frame_sink = frame_sink
        self.renderer = renderer
        self.writer = writer
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> TrackingRunResult:
        video_path, output_dir = self._validate_request()
        source_label = str(video_path) if video_path is not None else "injected frame source"
        algorithm = self._create_algorithm()
        tracker_label = (
            self.request.tracker
            if self.algorithm is None
            else type(algorithm).__name__
        )
        detector = self._create_detector()
        source = self._create_source(video_path)
        source_metadata = self._source_metadata(source)
        writer = self._create_writer(output_dir)
        tracking = RunObjectDetectionTrackingCommand(
            detection_model=detector,
            algorithm=algorithm,
            class_ids=self.request.track_class_ids,
        )

        emit(
            self.reporter,
            "info",
            f"Tracking '{source_label}' with {tracker_label}.",
        )
        processed = self._process_frames(
            source_label=source_label,
            source=source,
            writer=writer,
            tracking=tracking,
        )
        if processed.stopped_by_user and self.request.ground_truth:
            emit(
                self.reporter,
                "warning",
                "Tracking playback stopped early; skipped benchmarking because the "
                "prediction file does not cover the complete video.",
            )
            metrics_path, benchmark = None, None
        else:
            metrics_path, benchmark = self._run_benchmark(
                output_dir=output_dir,
                predictions_path=processed.output_path,
                frames_processed=processed.frames_processed,
            )
        replay = self._export_replay(
            output_dir=output_dir,
            predictions_path=processed.output_path,
            processed=processed,
            source_metadata=source_metadata,
            benchmark=benchmark,
            source_name=video_path.name if video_path is not None else source_label,
            tracker_label=tracker_label,
        )

        emit(
            self.reporter,
            "success",
            f"Tracked {processed.frames_processed} frame(s) and wrote "
            f"{writer.rows_written} MOT row(s) to {processed.output_path}. "
            f"Offline replay: {replay.html_path}.",
        )
        return TrackingRunResult(
            frames_processed=processed.frames_processed,
            tracks_written=writer.rows_written,
            output_path=processed.output_path,
            metrics_path=metrics_path,
            benchmark=benchmark,
            stopped_by_user=processed.stopped_by_user,
            replay_data_path=replay.data_path,
            replay_html_path=replay.html_path,
        )

    def _create_algorithm(self) -> TrackingAlgorithm:
        return self.algorithm or CreateTrackingAlgorithm(
            tracker=self.request.tracker,
            config_path=self.request.tracker_config,
        ).execute()

    def _create_detector(self) -> DetectionAdapter:
        return self.detector or CreateObjectDetector(
            ObjectDetectionRequest.from_config(self.request.detector_config())
        ).execute()

    def _create_source(self, video_path: Path | None) -> FrameSource:
        if self.frame_source is not None:
            return self.frame_source
        if video_path is None:
            raise RuntimeError("Validated video path is unavailable.")
        return OpenCVFrameSource(source="video", file_path=str(video_path))

    def _create_writer(self, output_dir: Path) -> MOTResultWriter:
        return self.writer or MOTResultWriter(
            output_dir=output_dir,
            overwrite=self.request.overwrite,
        )

    def _process_frames(
        self,
        *,
        source_label: str,
        source: FrameSource,
        writer: MOTResultWriter,
        tracking: RunObjectDetectionTrackingCommand,
    ) -> _ProcessedTrackingVideo:
        frames_processed = 0
        stopped_by_user = False
        frame_width = 0
        frame_height = 0
        writer_started = False
        finalized = False
        try:
            writer.start()
            writer_started = True
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                if frame.ndim < 2 or frame.shape[0] < 1 or frame.shape[1] < 1:
                    raise MLXUserError(
                        f"Tracking source '{source_label}' produced an invalid empty frame."
                    )
                frame_height = max(frame_height, int(frame.shape[0]))
                frame_width = max(frame_width, int(frame.shape[1]))
                result = tracking.execute(frame=frame)
                writer.write(result)
                frames_processed += 1
                if self.frame_sink is not None and self.renderer is not None:
                    rendered = self.renderer(frame, result)
                    if not self.frame_sink.show(rendered):
                        stopped_by_user = True
                        break
            if frames_processed == 0:
                raise MLXUserError(
                    f"Tracking source '{source_label}' did not produce any readable "
                    "frames. Check the source, file codec, and contents."
                )
            output_path = writer.finalize()
            finalized = True
            return _ProcessedTrackingVideo(
                frames_processed=frames_processed,
                output_path=output_path,
                stopped_by_user=stopped_by_user,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        finally:
            self._cleanup_session(
                source=source,
                tracking=tracking,
                writer=writer,
                abort_writer=writer_started and not finalized,
                active_error=sys.exc_info()[1],
            )

    def _cleanup_session(
        self,
        *,
        source: FrameSource,
        tracking: RunObjectDetectionTrackingCommand,
        writer: MOTResultWriter,
        abort_writer: bool,
        active_error: BaseException | None,
    ) -> None:
        cleanup_failures: list[tuple[str, Exception]] = []
        operations = [
            ("release frame source", source.release),
            ("reset tracker", tracking.reset),
        ]
        if self.frame_sink is not None:
            operations.append(("close tracking display", self.frame_sink.close))
        if abort_writer:
            operations.append(("remove incomplete output", writer.abort))
        # Attempt every cleanup operation while preserving the primary workflow error.
        for action, operation in operations:
            try:
                operation()
            except Exception as exc:
                cleanup_failures.append((action, exc))
        if not cleanup_failures:
            return
        message = "Tracking cleanup failed: " + "; ".join(
            f"{action}: {error}" for action, error in cleanup_failures
        )
        if active_error is None:
            raise MLXUserError(message) from cleanup_failures[0][1]
        if hasattr(active_error, "add_note"):
            active_error.add_note(message)
        emit(self.reporter, "warning", message)

    def _run_benchmark(
        self,
        *,
        output_dir: Path,
        predictions_path: Path,
        frames_processed: int,
    ) -> tuple[Path | None, TrackingBenchmarkResult | None]:
        if not self.request.ground_truth:
            return None, None
        metrics_path = output_dir / "metrics.json"
        benchmark = BenchmarkMOTTracking(
            ground_truth_path=Path(self.request.ground_truth).expanduser(),
            predictions_path=predictions_path,
            output_path=metrics_path,
            iou_threshold=self.request.benchmark_iou,
            processed_frames=frames_processed,
            overwrite=self.request.overwrite,
        ).execute()
        return metrics_path, benchmark

    def _source_metadata(self, source: FrameSource) -> FrameSourceMetadata:
        if not isinstance(source, MetadataFrameSource):
            return FrameSourceMetadata()
        try:
            return source.metadata()
        except Exception as exc:
            emit(
                self.reporter,
                "warning",
                f"Unable to read optional frame-source metadata: {exc}. "
                "Using decoded frame dimensions and 30 FPS for replay export.",
            )
        return FrameSourceMetadata()

    def _export_replay(
        self,
        *,
        output_dir: Path,
        predictions_path: Path,
        processed: _ProcessedTrackingVideo,
        source_metadata: FrameSourceMetadata,
        benchmark: TrackingBenchmarkResult | None,
        source_name: str,
        tracker_label: str,
    ) -> TrackingReplayResult:
        return ExportTrackingReplay(
            predictions_path=predictions_path,
            ground_truth_path=(
                Path(self.request.ground_truth).expanduser()
                if self.request.ground_truth
                else None
            ),
            output_dir=output_dir,
            frame_width=processed.frame_width,
            frame_height=processed.frame_height,
            frame_count=processed.frames_processed,
            fps=source_metadata.fps or 30.0,
            metrics=benchmark.to_dict() if benchmark is not None else None,
            run_metadata={
                "provider": self.request.provider,
                "tracker": tracker_label,
                "detector_confidence": self.request.confidence,
                "tracked_class_ids": list(self.request.track_class_ids),
                "benchmark_iou": self.request.benchmark_iou,
            },
            source_name=source_name,
            overwrite=self.request.overwrite,
        ).execute()

    def _validate_request(self) -> tuple[Path | None, Path]:
        if not self.request.file_path and self.frame_source is None:
            raise MLXUserError(
                "Tracking requires --file-path pointing to an input video."
            )
        if not self.request.output_path:
            raise MLXUserError(
                "Tracking requires --output pointing to a result directory."
            )
        video_path = (
            Path(self.request.file_path).expanduser()
            if self.request.file_path
            else None
        )
        if self.frame_source is None and video_path is not None and not video_path.is_file():
            raise MLXUserError(f"Tracking video file not found: {video_path}")
        if not 0.0 < self.request.benchmark_iou <= 1.0:
            raise MLXUserError("--benchmark-iou must be greater than 0 and at most 1.")
        if any(class_id < 0 for class_id in self.request.track_class_ids):
            raise MLXUserError("--track-class-id values must be zero or greater.")
        output_dir = Path(self.request.output_path).expanduser()
        output_paths = [output_dir / "tracks.txt"]
        output_paths.extend((output_dir / "replay.json", output_dir / "replay.html"))
        if self.request.ground_truth:
            output_paths.append(output_dir / "metrics.json")
        if not self.request.overwrite:
            existing = next((path for path in output_paths if path.exists()), None)
            if existing is not None:
                raise MLXUserError(
                    f"Tracking artifact already exists: {existing}. "
                    "Pass --overwrite to replace it."
                )
        if self.request.ground_truth:
            ground_truth_path = Path(self.request.ground_truth).expanduser()
            load_mot_ground_truth(ground_truth_path)
            require_motmetrics()
        return video_path, output_dir
