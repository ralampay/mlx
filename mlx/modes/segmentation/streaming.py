from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from mlx.core.exceptions import MLXUserError


class SegmentationFrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    def release(self) -> None:
        ...


class SegmentationFrameSink(Protocol):
    def show(self, frame: np.ndarray) -> bool:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class SegmentationStreamResult:
    frames_processed: int
    stopped_by_user: bool


class OpenCVSegmentationFrameSource:
    def __init__(
        self,
        *,
        source: str,
        camera_index: int = 0,
        file_path: str | None = None,
    ) -> None:
        if source == "camera":
            capture_source: int | str = camera_index
        elif source == "video":
            if not file_path:
                raise MLXUserError(
                    "Video inference requires --file-path pointing to the video file."
                )
            resolved = Path(file_path).expanduser()
            if not resolved.is_file():
                raise MLXUserError(f"Video file not found: {resolved}")
            capture_source = str(resolved)
        else:
            raise MLXUserError(f"Unsupported source type: {source}")
        self.capture = cv2.VideoCapture(capture_source)
        if not self.capture.isOpened():
            label = f"camera index {camera_index}" if source == "camera" else capture_source
            raise MLXUserError(f"Unable to open {label}. Check the source and permissions.")

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class OpenCVSegmentationFrameSink:
    def __init__(self, *, title: str, delay_ms: int) -> None:
        self.title = title
        self.delay_ms = delay_ms

    def show(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(self.delay_ms) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyAllWindows()
