from __future__ import annotations

from dataclasses import dataclass

from mlx.core.streaming import (
    FrameSink,
    FrameSource,
    FrameSourceMetadata,
    MetadataFrameSource,
    OpenCVFrameSink,
    OpenCVFrameSource,
)


@dataclass(frozen=True)
class StreamInferenceResult:
    frames_processed: int
    stopped_by_user: bool
    elapsed_seconds: float = 0.0
    average_fps: float = 0.0


__all__ = [
    "FrameSink",
    "FrameSource",
    "FrameSourceMetadata",
    "MetadataFrameSource",
    "OpenCVFrameSink",
    "OpenCVFrameSource",
    "StreamInferenceResult",
]
