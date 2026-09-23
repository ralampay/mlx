"""Dataset benchmark orchestration using existing tracking and MOT evaluation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import OpenCVFrameSource
from mlx.modes.object_detection.commands import CreateObjectDetector
from mlx.modes.object_detection.requests import ObjectDetectionRequest
from mlx.modes.object_detection.tracking.data import SequenceFrameSource, discover_tracking_sequences
from mlx.modes.object_detection.tracking.evaluation import require_motmetrics
from mlx.modes.object_detection.tracking.requests import TrackingBenchmarkRequest
from mlx.modes.object_detection.tracking.session import RunTrackingVideo
from mlx.modes.object_detection.tracking.video import CompileTrackingVideo, TrackingVideoSink, TrackingVideoWriter


@dataclass(frozen=True)
class TrackingDatasetBenchmarkResult:
    output_path: Path
    sequences: tuple[dict, ...]
    complete: bool


class BenchmarkTrackingDataset:
    def __init__(self, request: TrackingBenchmarkRequest, *, detector=None,
                 source_factory=SequenceFrameSource, compiler_factory=CompileTrackingVideo,
                 writer_factory=TrackingVideoWriter, session_factory=RunTrackingVideo,
                 display_factory=None, renderer_factory=None, reporter=None):
        self.request = request
        self.detector = detector
        self.source_factory = source_factory
        self.compiler_factory = compiler_factory
        self.writer_factory = writer_factory
        self.session_factory = session_factory
        self.display_factory = display_factory
        self.renderer_factory = renderer_factory
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> TrackingDatasetBenchmarkResult:
        request = self.request
        tracking = request.tracking
        if not tracking.output_path:
            raise MLXUserError("Tracking benchmark requires --output pointing to a result directory.")
        if not 0.0 < tracking.benchmark_iou <= 1.0:
            raise MLXUserError("--benchmark-iou must be greater than 0 and at most 1.")
        if any(class_id < 0 for class_id in tracking.track_class_ids):
            raise MLXUserError("--track-class-id values must be zero or greater.")
        if request.real_time_results and self.renderer_factory is None:
            raise MLXUserError("Visual tracking benchmarks require an injected trajectory renderer factory.")
        sequences = discover_tracking_sequences(Path(request.dataset_path).expanduser(), request.split)
        output = Path(tracking.output_path).expanduser().resolve()
        for sequence in sequences:
            if output.is_relative_to(sequence.root.resolve()) or sequence.root.resolve().is_relative_to(output):
                raise MLXUserError("Benchmark output must be separate from prepared sequence inputs.")
        if not tracking.overwrite:
            collisions = [output / name for name in ("summary.json", "summary.csv", "run_metadata.json")]
            collisions.extend(output / s.key for s in sequences)
            if any(p.exists() for p in collisions):
                raise MLXUserError(f"Benchmark artifacts already exist under '{output}'. Pass --overwrite to replace them.")
        require_motmetrics()
        detector = self.detector or CreateObjectDetector(ObjectDetectionRequest.from_config(tracking.detector_config())).execute()
        metadata = {
            "version": 1, "evaluation": "MLX MOT metrics (not official dataset protocols)",
            "settings": request, "complete": False,
            "model_sha256": sha256_file(tracking.model_path) if tracking.model_path and Path(tracking.model_path).is_file() else None,
            "inputs": [{"sequence": s.key, "manifest_sha256": sha256_file(s.root / "sequence.json")} for s in sequences],
        }
        write_json_atomic(output / "run_metadata.json", metadata)
        rows = []
        try:
            for sequence in sequences:
                emit(self.reporter, "info", f"Benchmarking {sequence.key} ({sequence.frame_count} frames).")
                row = self._run_sequence(sequence, output / sequence.key, detector)
                rows.append(row)
                self._write_summary(output, rows, False)
                if row["status"] != "complete":
                    break
        except Exception as exc:
            rows.append({"sequence": sequence.key, "status": "failed", "error": str(exc)})
            self._write_summary(output, rows, False)
            if isinstance(exc, MLXUserError):
                raise
            raise MLXUserError(f"Tracking benchmark failed for '{sequence.key}': {exc}") from exc
        complete = len(rows) == len(sequences) and all(r["status"] == "complete" for r in rows)
        self._write_summary(output, rows, complete)
        write_json_atomic(output / "run_metadata.json", {**metadata, "complete": complete})
        return TrackingDatasetBenchmarkResult(output, tuple(rows), complete)

    def _run_sequence(self, sequence, output, detector):
        request = self.request
        writer = source = None
        compiled = annotated = None
        sink = renderer = None
        try:
            # Only remove artifacts owned by this command; unrelated files survive.
            # In particular, a cancelled overwrite must not retain old metrics.
            if request.tracking.overwrite:
                for name in ("tracks.txt", "tracks.jsonl", "metrics.json", "replay.json", "replay.html", "compiled.avi", "annotated.mp4"):
                    (output / name).unlink(missing_ok=True)
            if request.real_time_results:
                emit(self.reporter, "info", f"Compiling and verifying lossless video for {sequence.key}.")
                compiled = self.compiler_factory(
                    source_factory=lambda: self.source_factory(sequence), metadata=sequence.metadata(),
                    output_path=output / "compiled.avi",
                ).execute()
                source = OpenCVFrameSource(source="video", file_path=str(compiled))
                annotated = output / "annotated.mp4"
                writer = self.writer_factory(annotated, fps=sequence.fps, width=sequence.width, height=sequence.height)
                display = self.display_factory() if request.tracking.display and self.display_factory else None
                sink = TrackingVideoSink(writer, display)
                renderer = self.renderer_factory()
            else:
                source = self.source_factory(sequence)
            result = self.session_factory(
                replace(request.tracking, file_path=None, output_path=str(output), ground_truth=str(sequence.ground_truth)),
                detector=detector, frame_source=source, frame_sink=sink, renderer=renderer, reporter=self.reporter,
            ).execute()
            if not result.stopped_by_user and result.frames_processed != sequence.frame_count:
                raise MLXUserError(f"Sequence '{sequence.key}' ended before all expected frames were evaluated.")
            if writer is not None:
                writer.finalize()
            return {
                "sequence": sequence.key, "status": "cancelled" if result.stopped_by_user else "complete",
                "frames_processed": result.frames_processed,
                "compiled_video": str(compiled) if compiled else None,
                "annotated_video": str(annotated) if annotated else None,
                "result": result,
                **(result.benchmark.to_dict() if result.benchmark else {}),
            }
        except BaseException:
            if writer is not None:
                try:
                    writer.abort()
                except OSError as exc:
                    emit(self.reporter, "warning", f"Unable to remove incomplete video: {exc}")
            raise
        finally:
            active_error = sys.exc_info()[1]
            cleanup_errors = []
            for resource in (source, sink):
                if resource is None:
                    continue
                try:
                    resource.release() if resource is source else resource.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                message = f"Benchmark resource cleanup failed: {cleanup_errors}"
                if active_error is None:
                    raise MLXUserError(message) from cleanup_errors[0]
                emit(self.reporter, "warning", message)

    @staticmethod
    def _write_summary(output, rows, complete):
        write_json_atomic(output / "summary.json", {"complete": complete, "sequences": rows})
        columns = ["sequence", "status", "frames_processed", "mota", "motp", "idf1", "precision", "recall",
                   "ground_truth_objects", "predictions", "matches", "false_positives", "misses", "id_switches",
                   "compiled_video", "annotated_video", "error"]
        write_csv(output / "summary.csv", rows, fieldnames=columns)
