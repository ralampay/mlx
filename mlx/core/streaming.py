from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from mlx.core.exceptions import MLXUserError


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class FrameSink(Protocol):
    def show(self, frame: np.ndarray) -> bool: ...

    def close(self) -> None: ...


class NullFrameSink:
    """Headless sink that consumes rendered frames without terminal or GUI output."""

    def show(self, frame: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        return None


class OpenCVFrameSink:
    """Shared lazy OpenCV display that stops on q or Escape."""

    def __init__(self, *, title: str, delay_ms: int,
                 max_size: tuple[int, int] | None = None) -> None:
        if max_size is not None and (
            len(max_size) != 2 or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in max_size
            )
        ):
            raise MLXUserError("Display max_size must contain positive integer width and height.")
        font_directory = os.environ.get("QT_QPA_FONTDIR")
        try:
            import cv2
        except ImportError as exc:
            raise MLXUserError(
                "OpenCV is required for stream display. Install opencv-python and try again."
            ) from exc
        from mlx.core.opencv_display import configure_opencv_display

        configure_opencv_display(os.environ, font_directory=font_directory)
        self._cv2 = cv2
        self.title = title
        self.delay_ms = delay_ms
        self.max_size = max_size
        self._window_initialized = False

    def show(self, frame: np.ndarray) -> bool:
        from mlx.core.opencv_display import opencv_display_environment

        display_frame = self._display_frame(frame)
        with opencv_display_environment(os.environ):
            if self.max_size is not None and not self._window_initialized:
                self._cv2.namedWindow(
                    self.title,
                    self._cv2.WINDOW_NORMAL | self._cv2.WINDOW_KEEPRATIO
                    | self._cv2.WINDOW_GUI_NORMAL,
                )
                height, width = display_frame.shape[:2]
                self._cv2.resizeWindow(self.title, width, height)
                self._window_initialized = True
            self._cv2.imshow(self.title, display_frame)
        key = self._cv2.waitKey(self.delay_ms) & 0xFF
        return key not in (ord("q"), 27)

    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.max_size is None:
            return frame
        height, width = frame.shape[:2]
        scale = min(self.max_size[0] / width, self.max_size[1] / height, 1.0)
        if scale == 1.0:
            return frame
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return self._cv2.resize(frame, size, interpolation=self._cv2.INTER_AREA)

    def close(self) -> None:
        self._cv2.destroyAllWindows()
        self._window_initialized = False


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

    def read(self) -> tuple[bool, np.ndarray | None]:
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
    "FrameSink",
    "FrameSource",
    "FrameSourceMetadata",
    "MetadataFrameSource",
    "NullFrameSink",
    "OpenCVFrameSink",
    "OpenCVFrameSource",
]
