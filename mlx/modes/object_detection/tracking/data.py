"""Portable prepared-sequence manifests and frame selection."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import FrameSourceMetadata, OpenCVFrameSource
from mlx.modes.object_detection.tracking.evaluation import load_mot_ground_truth


@dataclass(frozen=True)
class TrackingSequence:
    root: Path
    dataset: str
    split: str
    name: str
    media: Path
    kind: str
    ground_truth: Path
    fps: float
    width: int
    height: int
    frame_indices: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.name}"

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    def metadata(self) -> FrameSourceMetadata:
        return FrameSourceMetadata(self.width, self.height, self.fps, self.frame_count)


def numeric_images(directory: Path) -> tuple[Path, ...]:
    images = tuple(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images or any(not p.stem.isdigit() for p in images):
        raise MLXUserError(f"Expected numerically named image frames in '{directory}'.")
    images = tuple(sorted(images, key=lambda p: int(p.stem)))
    numbers = [int(p.stem) for p in images]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise MLXUserError(f"Duplicate or missing frame numbers in '{directory}'.")
    return images


def _relative_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest paths must remain inside the sequence directory.")
    if not path.exists():
        raise ValueError(f"Missing sequence input: {path}")
    return path


def load_tracking_sequence(path: Path) -> TrackingSequence:
    try:
        data = json.loads(path.read_text())
        if data["version"] != 1:
            raise ValueError("Unsupported sequence manifest version.")
        for field in ("dataset", "split", "name"):
            value = data[field]
            if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
                raise ValueError(f"Invalid sequence {field}.")
        indices = tuple(data["frame_indices"])
        if not indices or any(type(i) is not int or i < 0 for i in indices) or tuple(sorted(set(indices))) != indices:
            raise ValueError("Source frame indices must be unique, increasing, nonnegative integers.")
        fps = float(data["fps"])
        width, height = int(data["width"]), int(data["height"])
        if not math.isfinite(fps) or fps <= 0 or width < 1 or height < 1:
            raise ValueError("FPS and dimensions must be positive.")
        kind = data["kind"]
        if kind not in {"images", "video"}:
            raise ValueError("Sequence kind must be images or video.")
        media = _relative_file(path.parent, data["media"])
        gt = _relative_file(path.parent, data["ground_truth"])
        if kind == "images":
            if not media.is_dir() or indices != tuple(range(len(numeric_images(media)))):
                raise ValueError("Image sequence frame count does not match its manifest.")
        elif not media.is_file():
            raise ValueError("Sequence video is not a file.")
        records = load_mot_ground_truth(gt)
        if max(row.frame_id for row in records) > len(indices):
            raise ValueError("Ground truth extends beyond the selected frames.")
        return TrackingSequence(path.parent, data["dataset"], data["split"], data["name"],
                                media, kind, gt, fps, width, height, indices)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise MLXUserError(f"Invalid tracking manifest '{path}': {exc}") from exc


def discover_tracking_sequences(root: Path, split: str | None = None) -> tuple[TrackingSequence, ...]:
    if not root.is_dir():
        raise MLXUserError(f"Tracking benchmark directory not found: {root}")
    paths = [root / "sequence.json"] if (root / "sequence.json").is_file() else sorted(root.rglob("sequence.json"))
    sequences = tuple(load_tracking_sequence(p) for p in paths)
    sequences = tuple(s for s in sequences if split is None or s.split == split)
    if not sequences:
        raise MLXUserError(f"No prepared tracking sequences found under '{root}' for split {split!r}.")
    if len({s.key for s in sequences}) != len(sequences):
        raise MLXUserError("Duplicate dataset/split/sequence identifiers in benchmark inputs.")
    return tuple(sorted(sequences, key=lambda s: s.key))


class SequenceFrameSource:
    """Decode precisely the manifest's evaluation frames, without random seeking."""

    def __init__(self, sequence: TrackingSequence):
        self.sequence = sequence
        self._images = numeric_images(sequence.media) if sequence.kind == "images" else ()
        self._video = OpenCVFrameSource(source="video", file_path=str(sequence.media)) if sequence.kind == "video" else None
        self._position = 0
        self._source_position = 0

    def metadata(self):
        return self.sequence.metadata()

    def read(self):
        if self._position == self.sequence.frame_count:
            return False, None
        source_index = self.sequence.frame_indices[self._position]
        if self._video is None:
            import cv2
            frame = cv2.imread(str(self._images[source_index]))
        else:
            frame = None
            while self._source_position <= source_index:
                ok, frame = self._video.read()
                if not ok:
                    raise MLXUserError(f"Sequence '{self.sequence.key}' cannot decode source frame {self._source_position}.")
                self._source_position += 1
        if frame is None or frame.shape != (self.sequence.height, self.sequence.width, 3):
            raise MLXUserError(f"Sequence '{self.sequence.key}' has an unreadable or mismatched frame at {source_index}.")
        self._position += 1
        return True, frame

    def release(self):
        if self._video is not None:
            self._video.release()
