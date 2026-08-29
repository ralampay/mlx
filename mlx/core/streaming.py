from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from mlx.core.exceptions import MLXUserError


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FrameSourceMetadata:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None


@runtime_checkable
class MetadataFrameSource(Protocol):
    def metadata(self) -> FrameSourceMetadata: ...


class OpenCVFrameSource:
    """Shared lazy OpenCV camera/video decoder."""

    def __init__(
        self,
        *,
        source: str,
        camera_index: int = 0,
        file_path: str | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise MLXUserError(
                "OpenCV is required for stream inference. Install opencv-python and try again."
            ) from exc

        if source == "camera":
            capture = cv2.VideoCapture(camera_index)
            label = f"camera index {camera_index}"
        elif source == "video":
            if not file_path:
                raise MLXUserError(
                    "Video inference requires --file-path pointing to a video file."
                )
            video_path = Path(file_path).expanduser()
            if not video_path.is_file():
                raise MLXUserError(f"Video file not found: {video_path}")
            capture = cv2.VideoCapture(str(video_path))
            label = f"video file {video_path}"
        else:
            raise MLXUserError(f"Unsupported stream source '{source}'.")

        if not capture.isOpened():
            capture.release()
            raise MLXUserError(
                f"Unable to open {label}. Check the source and permissions."
            )
        self._capture = capture
        self._cv2 = cv2

    def read(self) -> tuple[bool, np.ndarray]:
        return self._capture.read()

    def release(self) -> None:
        self._capture.release()

    def metadata(self) -> FrameSourceMetadata:
        return FrameSourceMetadata(
            width=_positive_int_or_none(
                self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)
            ),
            height=_positive_int_or_none(
                self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)
            ),
            fps=_positive_float_or_none(self._capture.get(self._cv2.CAP_PROP_FPS)),
            frame_count=_positive_int_or_none(
                self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT)
            ),
        )


def _positive_int_or_none(value: float) -> int | None:
    return int(round(value)) if value > 0 else None


def _positive_float_or_none(value: float) -> float | None:
    return float(value) if value > 0 else None


__all__ = [
    "FrameSource",
    "FrameSourceMetadata",
    "MetadataFrameSource",
    "OpenCVFrameSource",
]
