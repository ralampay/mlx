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


__all__ = [
    "FrameSink",
    "FrameSource",
    "FrameSourceMetadata",
    "MetadataFrameSource",
    "OpenCVFrameSink",
    "OpenCVFrameSource",
    "StreamInferenceResult",
]
