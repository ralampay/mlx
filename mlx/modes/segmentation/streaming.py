from __future__ import annotations

from dataclasses import dataclass

from mlx.core.streaming import (
    FrameSink as SegmentationFrameSink,
    FrameSource as SegmentationFrameSource,
    OpenCVFrameSink as OpenCVSegmentationFrameSink,
    OpenCVFrameSource,
)


@dataclass(frozen=True)
class SegmentationStreamResult:
    frames_processed: int
    stopped_by_user: bool


class OpenCVSegmentationFrameSource(OpenCVFrameSource):
    """Shared decoder retaining the historic public capture attribute."""

    @property
    def capture(self):
        return self._capture

    @capture.setter
    def capture(self, value):
        self._capture = value


__all__ = [
    "OpenCVSegmentationFrameSink",
    "OpenCVSegmentationFrameSource",
    "SegmentationFrameSink",
    "SegmentationFrameSource",
    "SegmentationStreamResult",
]
