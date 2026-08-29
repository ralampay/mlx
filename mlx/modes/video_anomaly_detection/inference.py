from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import FrameSink, MetadataFrameSource, OpenCVFrameSource
from mlx.modes.video_anomaly_detection.artifacts import (
    load_video_anomaly_checkpoint,
    write_csv,
    write_json,
    write_jsonl,
)
from mlx.modes.video_anomaly_detection.data import build_bgr_frame_transform
from mlx.modes.video_anomaly_detection.requests import InferVideoAnomalyRequest


@dataclass(frozen=True)
class VideoAnomalyInferenceResult:
    input_path: str | None
    output_path: str
    anomaly_detected: bool
    windows_scored: int
    anomalous_windows: int
    max_anomaly_score: float | None
    threshold: float
    frames_displayed: int
    stopped_by_user: bool
    predictions_path: str
    predictions_csv_path: str
    summary_path: str
    predictions: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "anomaly_detected": self.anomaly_detected,
            "windows_scored": self.windows_scored,
            "anomalous_windows": self.anomalous_windows,
            "max_anomaly_score": self.max_anomaly_score,
            "threshold": self.threshold,
            "frames_displayed": self.frames_displayed,
            "stopped_by_user": self.stopped_by_user,
            "predictions_path": self.predictions_path,
            "predictions_csv_path": self.predictions_csv_path,
        }


class InferVideoAnomaly:
    def __init__(
        self,
        request: InferVideoAnomalyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        checkpoint_loader=load_video_anomaly_checkpoint,
        frame_source=None,
        frame_source_factory=OpenCVFrameSource,
        frame_transform=None,
        frame_sink: FrameSink | None = None,
        frame_renderer: Callable[
            [np.ndarray, Mapping[str, Any] | None, int, int], np.ndarray
        ]
        | None = None,
    ) -> None:
        if (frame_sink is None) != (frame_renderer is None):
            raise ValueError(
                "Video anomaly display requires both a frame sink and frame renderer."
            )
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.checkpoint_loader = checkpoint_loader
        self.frame_source = frame_source
        self.frame_source_factory = frame_source_factory
        self.frame_transform = frame_transform
        self.frame_sink = frame_sink
        self.frame_renderer = frame_renderer

    def execute(self) -> VideoAnomalyInferenceResult:
        config = self.request.to_config()
        if not config.get("model_path"):
            raise MLXUserError("Video inference requires --model-path.")
        model, checkpoint, stored = self.checkpoint_loader(
            config["model_path"],
            device=str(config["device"]),
            model_name=config.get("model"),
        )
        try:
            threshold = float(checkpoint["svdd_threshold"])
        except (KeyError, TypeError, ValueError):
            raise MLXUserError(
                "Video inference requires a checkpoint with a normal-validation calibrated threshold."
            ) from None
        if not np.isfinite(threshold):
            raise MLXUserError(
                "Video inference requires a checkpoint with a normal-validation calibrated threshold."
            )
        if int(config["batch_size"]) < 1:
            raise MLXUserError("Video inference batch size must be a positive integer.")
        span = (stored["clip_length"] - 1) * stored["frame_stride"] + 1
        frame_transform = self.frame_transform or build_bgr_frame_transform(
            height=stored["height"], width=stored["width"]
        )
        source = self.frame_source or self.frame_source_factory(
            source="video", file_path=config.get("file_path")
        )
        buffered: deque[tuple[int, torch.Tensor]] = deque(maxlen=span)
        pending: list[tuple[list[int], torch.Tensor]] = []
        records: list[dict[str, Any]] = []
        frame_index = 0
        frames_displayed = 0
        stopped_by_user = False
        try:
            fps = None
            if isinstance(source, MetadataFrameSource):
                fps = source.metadata().fps
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                buffered.append(
                    (
                        frame_index,
                        frame_transform(frame),
                    )
                )
                frame_index += 1
                if len(buffered) < span:
                    if not self._display_frame(
                        frame,
                        prediction=None,
                        frames_buffered=len(buffered),
                        frames_required=span,
                    ):
                        frames_displayed += 1
                        stopped_by_user = True
                        break
                    if self.frame_sink is not None:
                        frames_displayed += 1
                    continue

                selected = list(buffered)[:: stored["frame_stride"]]
                indices, tensors = zip(*selected, strict=True)
                window = (list(indices), torch.stack(tensors))
                if self.frame_sink is not None:
                    prediction = self._score_batch(
                        model,
                        [window],
                        str(config["device"]),
                        threshold,
                        fps,
                    )[0]
                    records.append(prediction)
                    if not self._display_frame(
                        frame,
                        prediction=prediction,
                        frames_buffered=span,
                        frames_required=span,
                    ):
                        frames_displayed += 1
                        stopped_by_user = True
                        break
                    frames_displayed += 1
                else:
                    pending.append(window)
                    if len(pending) >= int(config["batch_size"]):
                        records.extend(
                            self._score_batch(
                                model,
                                pending,
                                str(config["device"]),
                                threshold,
                                fps,
                            )
                        )
                        pending.clear()
            if pending and not stopped_by_user:
                records.extend(
                    self._score_batch(
                        model,
                        pending,
                        str(config["device"]),
                        threshold,
                        fps,
                    )
                )
        finally:
            source.release()
            if self.frame_sink is not None:
                self.frame_sink.close()
        if not records and not stopped_by_user:
            raise MLXUserError(
                f"Video produced no complete clip window; at least {span} decoded frames are required."
            )
        output_dir = Path(config.get("output_path") or "video-anomaly-inference").expanduser()
        predictions_path = output_dir / "predictions.jsonl"
        predictions_csv_path = output_dir / "predictions.csv"
        summary_path = output_dir / "summary.json"
        anomalous_windows = sum(bool(record["is_anomaly"]) for record in records)
        result = VideoAnomalyInferenceResult(
            input_path=config.get("file_path"),
            output_path=str(output_dir),
            anomaly_detected=anomalous_windows > 0,
            windows_scored=len(records),
            anomalous_windows=anomalous_windows,
            max_anomaly_score=(
                max(float(record["anomaly_score"]) for record in records)
                if records
                else None
            ),
            threshold=threshold,
            frames_displayed=frames_displayed,
            stopped_by_user=stopped_by_user,
            predictions_path=str(predictions_path),
            predictions_csv_path=str(predictions_csv_path),
            summary_path=str(summary_path),
            predictions=tuple(records),
        )
        write_jsonl(predictions_path, records)
        write_csv(predictions_csv_path, records)
        write_json(summary_path, result.summary())
        if result.anomaly_detected:
            verdict = "Anomaly detected"
        elif not records:
            verdict = "Display stopped before a complete temporal window was scored"
        else:
            verdict = "No anomaly detected"
        emit(
            self.reporter,
            "warning" if result.anomaly_detected else "success",
            (
                f"{verdict}: {anomalous_windows} of {len(records)} temporal windows "
                f"exceeded the threshold. Results written to {output_dir}"
            ),
            payload={
                "event": "video_anomaly_inference",
                **result.summary(),
            },
        )
        return result

    def _display_frame(
        self,
        frame: np.ndarray,
        *,
        prediction: Mapping[str, Any] | None,
        frames_buffered: int,
        frames_required: int,
    ) -> bool:
        if self.frame_sink is None or self.frame_renderer is None:
            return True
        rendered = self.frame_renderer(
            frame,
            prediction,
            frames_buffered,
            frames_required,
        )
        return self.frame_sink.show(rendered)

    @staticmethod
    @torch.no_grad()
    def _score_batch(model, pending, device: str, threshold: float, fps: float | None):
        model.eval()
        clips = torch.stack([clip for _, clip in pending]).to(device)
        scores = model(clips).anomaly_score.cpu().tolist()
        if any(not np.isfinite(float(score)) for score in scores):
            raise MLXUserError(
                "Video anomaly inference produced a non-finite anomaly score. "
                "Check that the checkpoint and input preprocessing are compatible."
            )
        rows = []
        for (indices, _), score in zip(pending, scores, strict=True):
            start, end = indices[0], indices[-1]
            rows.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "frame_indices": indices,
                    "start_time_seconds": start / fps if fps else None,
                    "end_time_seconds": end / fps if fps else None,
                    "anomaly_score": float(score),
                    "threshold": threshold,
                    "is_anomaly": bool(score > threshold),
                }
            )
        return rows


__all__ = ["InferVideoAnomaly", "VideoAnomalyInferenceResult"]
