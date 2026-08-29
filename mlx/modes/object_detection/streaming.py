from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import (
    FrameSource,
    FrameSourceMetadata,
    MetadataFrameSource,
    OpenCVFrameSource,
)


class FrameSink(Protocol):
    def show(self, frame: np.ndarray) -> bool:
        """Render a frame and return false when the session should stop."""

        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class StreamInferenceResult:
    frames_processed: int
    stopped_by_user: bool


class OpenCVFrameSink:
    def __init__(self, *, title: str, delay_ms: int) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise MLXUserError(
                "OpenCV is required for stream display. Install opencv-python and try again."
            ) from exc
        self._cv2 = cv2
        self.title = title
        self.delay_ms = delay_ms

    def show(self, frame: np.ndarray) -> bool:
        self._cv2.imshow(self.title, frame)
        key = self._cv2.waitKey(self.delay_ms) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        self._cv2.destroyAllWindows()
