from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mlx.core.exceptions import MLXUserError
from mlx.core.datasets import resolve_split_dataset_root
from mlx.modes.video_anomaly_detection.clips import ClipWindow, window_start_indices

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required for video anomaly datasets.") from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def video_anomaly_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        required_paths=("train/normal", "val/normal"),
        dataset_label="video-anomaly dataset",
    )


def build_frame_transform(*, height: int, width: int):
    if height < 1 or width < 1:
        raise MLXUserError("--height and --width must be positive.")
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_bgr_frame_transform(*, height: int, width: int):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise MLXUserError("OpenCV is required for video frame conversion.") from exc
    image_transform = build_frame_transform(height=height, width=width)

    def transform(frame) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image_transform(Image.fromarray(rgb))

    return transform


def transform_bgr_frame(frame, *, height: int, width: int) -> torch.Tensor:
    return build_bgr_frame_transform(height=height, width=width)(frame)


class VideoClipDataset(Dataset):
    """Deterministic fixed-window dataset over label-organized frame sequences."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        clip_length: int,
        frame_stride: int,
        height: int,
        width: int,
        normal_only: bool = False,
        window_stride: int = 1,
        transform=None,
    ) -> None:
        if clip_length < 1:
            raise MLXUserError("--clip-length must be at least 1.")
        if frame_stride < 1:
            raise MLXUserError("--frame-stride must be at least 1.")
        if window_stride < 1:
            raise MLXUserError("Window stride must be at least 1.")
        self.dataset_path = Path(dataset_path).expanduser()
        self.split = split
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.window_stride = window_stride
        self.height = height
        self.width = width
        self.transform = transform or build_frame_transform(height=height, width=width)
        self.root_dir = self._resolve_split_dir()
        self.windows: list[ClipWindow] = []

        anomaly_dir = self.root_dir / "anomaly"
        if normal_only and _contains_sources(anomaly_dir):
            raise MLXUserError(
                f"Anomalous samples were found in the normal-only '{split}' split: {anomaly_dir}"
            )

        labels = (("normal", 0),) if normal_only else (("normal", 0), ("anomaly", 1))
        for label_name, ground_truth in labels:
            self._index_label(self.root_dir / label_name, ground_truth)

        if not self.windows:
            expected_span = (clip_length - 1) * frame_stride + 1
            raise MLXUserError(
                f"No complete {clip_length}-frame clip windows (source span {expected_span}) "
                f"were found under: {self.root_dir}"
            )

    def _resolve_split_dir(self) -> Path:
        if not self.dataset_path.exists():
            raise MLXUserError(f"Dataset directory not found: {self.dataset_path}")
        candidate = self.dataset_path / self.split
        root = candidate if candidate.is_dir() else self.dataset_path
        if not (root / "normal").is_dir():
            raise MLXUserError(
                f"Dataset split must contain a normal directory: {root / 'normal'}"
            )
        return root

    def _index_label(self, label_dir: Path, ground_truth: int) -> None:
        if not label_dir.is_dir():
            return
        video_files = sorted(
            path for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if video_files:
            raise MLXUserError(
                "Video-file datasets are not yet decoded by the training dataset. "
                "Extract each video into a frame-sequence directory; direct video files are supported by infer-video."
            )
        for source_dir in sorted(path for path in label_dir.iterdir() if path.is_dir()):
            frame_paths = sorted(
                path for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            starts = window_start_indices(
                len(frame_paths),
                clip_length=self.clip_length,
                frame_stride=self.frame_stride,
                window_stride=self.window_stride,
            )
            for start in starts:
                positions = tuple(
                    start + offset * self.frame_stride
                    for offset in range(self.clip_length)
                )
                paths = tuple(frame_paths[position] for position in positions)
                indices = tuple(_frame_index(path, position) for path, position in zip(paths, positions, strict=True))
                self.windows.append(
                    ClipWindow(
                        source=f"{label_dir.name}/{source_dir.name}",
                        frame_paths=tuple(str(path) for path in paths),
                        frame_indices=indices,
                        start_frame=indices[0],
                        end_frame=indices[-1],
                        ground_truth=ground_truth,
                    )
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        window = self.windows[index]
        frames = []
        for frame_path in window.frame_paths:
            try:
                with Image.open(frame_path) as image:
                    frames.append(self.transform(image.convert("RGB")))
            except (OSError, ValueError) as exc:
                raise MLXUserError(f"Cannot read video frame '{frame_path}': {exc}") from exc
        metadata: dict[str, Any] = {
            "source": window.source,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "frame_indices": list(window.frame_indices),
        }
        return (
            torch.stack(frames),
            torch.tensor(window.ground_truth, dtype=torch.long),
            metadata,
        )


def collate_clip_samples(batch):
    clips, labels, metadata = zip(*batch, strict=True)
    return torch.stack(clips), torch.stack(labels), list(metadata)


def _contains_sources(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        (path.is_dir() and any(child.is_file() for child in path.iterdir()))
        or (path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
        for path in directory.iterdir()
    )


def _frame_index(path: Path, fallback_position: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback_position


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "VideoClipDataset",
    "build_bgr_frame_transform",
    "build_frame_transform",
    "collate_clip_samples",
    "transform_bgr_frame",
]
