from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import MetadataFrameSource, OpenCVFrameSource
from mlx.modes.video_anomaly_detection.artifacts import (
    load_video_anomaly_checkpoint,
    write_csv,
    write_jsonl,
)
from mlx.modes.video_anomaly_detection.data import build_bgr_frame_transform
from mlx.modes.video_anomaly_detection.requests import InferVideoAnomalyRequest


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
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.checkpoint_loader = checkpoint_loader
        self.frame_source = frame_source
        self.frame_source_factory = frame_source_factory
        self.frame_transform = frame_transform

    def execute(self) -> list[dict[str, Any]]:
        config = self.request.to_config()
        if not config.get("model_path"):
            raise MLXUserError("Video inference requires --model-path.")
        model, checkpoint, stored = self.checkpoint_loader(
            config["model_path"],
            device=str(config["device"]),
            model_name=config.get("model"),
        )
        threshold = checkpoint.get("svdd_threshold")
        if threshold is None or not np.isfinite(float(threshold)):
            raise MLXUserError(
                "Video inference requires a checkpoint with a normal-validation calibrated threshold."
            )
        source = self.frame_source or self.frame_source_factory(
            source="video", file_path=config.get("file_path")
        )
        fps = None
        if isinstance(source, MetadataFrameSource):
            fps = source.metadata().fps
        span = (stored["clip_length"] - 1) * stored["frame_stride"] + 1
        frame_transform = self.frame_transform or build_bgr_frame_transform(
            height=stored["height"], width=stored["width"]
        )
        buffered: deque[tuple[int, torch.Tensor]] = deque(maxlen=span)
        pending: list[tuple[list[int], torch.Tensor]] = []
        records: list[dict[str, Any]] = []
        frame_index = 0
        try:
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
                if len(buffered) == span:
                    selected = list(buffered)[:: stored["frame_stride"]]
                    indices, tensors = zip(*selected, strict=True)
                    pending.append((list(indices), torch.stack(tensors)))
                    if len(pending) >= int(config["batch_size"]):
                        records.extend(
                            self._score_batch(
                                model,
                                pending,
                                str(config["device"]),
                                float(threshold),
                                fps,
                            )
                        )
                        pending.clear()
            if pending:
                records.extend(
                    self._score_batch(
                        model,
                        pending,
                        str(config["device"]),
                        float(threshold),
                        fps,
                    )
                )
        finally:
            source.release()
        if not records:
            raise MLXUserError(
                f"Video produced no complete clip window; at least {span} decoded frames are required."
            )
        output_dir = Path(config.get("output_path") or "video-anomaly-inference").expanduser()
        write_jsonl(output_dir / "predictions.jsonl", records)
        write_csv(output_dir / "predictions.csv", records)
        emit(
            self.reporter,
            "success",
            f"Wrote {len(records)} temporal anomaly predictions to {output_dir}",
            payload={
                "event": "video_anomaly_inference",
                "predictions": len(records),
                "output": str(output_dir),
            },
        )
        return records

    @staticmethod
    @torch.no_grad()
    def _score_batch(model, pending, device: str, threshold: float, fps: float | None):
        model.eval()
        clips = torch.stack([clip for _, clip in pending]).to(device)
        scores = model(clips).anomaly_score.cpu().tolist()
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


__all__ = ["InferVideoAnomaly"]
