"""Video encoding at the tracking integration boundary."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import OpenCVFrameSource


class TrackingVideoWriter:
    def __init__(self, path: Path, *, fps: float, width: int, height: int, lossless: bool = False):
        import cv2
        self.path = path
        self.temporary = path.with_name(f".{path.stem}.partial{path.suffix}")
        self.size = (width, height)
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        # FFV1 preserves pixel values for benchmark parity. MP4 is for presentation.
        self._writer = cv2.VideoWriter(str(self.temporary), cv2.VideoWriter_fourcc(*("FFV1" if lossless else "mp4v")), fps, self.size)
        if not self._writer.isOpened():
            self.abort()
            raise MLXUserError(f"Unable to create video '{path}'. Check OpenCV codec support and output permissions.")

    def write(self, frame):
        if frame.shape[:2] != (self.size[1], self.size[0]):
            raise MLXUserError(f"Video frame dimensions changed while writing '{self.path}'.")
        self._writer.write(frame)
        self.count += 1

    def finalize(self):
        self.finish_encoding()
        if self.count == 0 or not self.temporary.is_file() or self.temporary.stat().st_size == 0:
            raise MLXUserError(f"Video encoder produced no output for '{self.path}'.")
        decoded = OpenCVFrameSource(source="video", file_path=str(self.temporary))
        try:
            metadata = decoded.metadata()
            ok, frame = decoded.read()
            if not ok or metadata.frame_count != self.count or frame.shape[:2] != (self.size[1], self.size[0]):
                raise MLXUserError(f"Encoded video is incomplete or has changed dimensions: {self.path}")
        finally:
            decoded.release()
        self.temporary.replace(self.path)
        return self.path

    def finish_encoding(self):
        self._writer.release()

    def abort(self):
        self._writer.release()
        self.temporary.unlink(missing_ok=True)


class CompileTrackingVideo:
    def __init__(self, *, source_factory, metadata, output_path: Path, writer_factory=TrackingVideoWriter):
        self.source_factory = source_factory
        self.metadata = metadata
        self.output_path = output_path
        self.writer_factory = writer_factory

    def execute(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        info = self.metadata
        estimate = info.width * info.height * 3 * info.frame_count
        if shutil.disk_usage(self.output_path.parent).free < estimate:
            raise MLXUserError(f"Insufficient space for lossless video '{self.output_path}'; allow up to {estimate} bytes.")
        source = self.source_factory()
        writer = None
        try:
            writer = self.writer_factory(self.output_path, fps=info.fps, width=info.width, height=info.height, lossless=True)
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                writer.write(frame)
            if writer.count != info.frame_count:
                raise MLXUserError("Compiled frame count does not match the tracking manifest.")
            # Validate the staged video against original decoded pixels before publishing.
            writer.finish_encoding()
            self._validate(writer.temporary)
            return writer.finalize()
        except BaseException as exc:
            if writer is not None:
                try:
                    writer.abort()
                except OSError as cleanup_exc:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"Unable to remove incomplete compiled video: {cleanup_exc}")
            if isinstance(exc, (MLXUserError, KeyboardInterrupt, SystemExit)):
                raise
            raise MLXUserError(f"Unable to compile tracking video '{self.output_path}': {exc}") from exc
        finally:
            active_error = sys.exc_info()[1]
            try:
                source.release()
            except Exception as exc:
                if active_error is None:
                    raise MLXUserError(f"Unable to release compilation source: {exc}") from exc
                if hasattr(active_error, "add_note"):
                    active_error.add_note(f"Unable to release compilation source: {exc}")

    def _validate(self, path):
        import numpy as np
        source = self.source_factory()
        decoded = None
        try:
            decoded = OpenCVFrameSource(source="video", file_path=str(path))
            for _ in range(self.metadata.frame_count):
                original_ok, original = source.read()
                decoded_ok, frame = decoded.read()
                if not original_ok or not decoded_ok or not np.array_equal(original, frame):
                    raise MLXUserError(f"Lossless video verification failed: {path}")
            if decoded.read()[0]:
                raise MLXUserError(f"Compiled video contains unexpected extra frames: {path}")
        finally:
            source.release()
            if decoded is not None:
                decoded.release()


class TrackingVideoSink:
    """Save every rendered frame and optionally forward it to a live display."""

    def __init__(self, writer, display=None):
        self.writer = writer
        self.display = display
        self._closed = False

    def show(self, frame):
        self.writer.write(frame)
        return self.display.show(frame) if self.display is not None else True

    def close(self):
        if self.display is not None and not self._closed:
            self._closed = True
            self.display.close()
