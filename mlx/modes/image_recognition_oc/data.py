from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mlx.core.datasets import resolve_split_dataset_root
from mlx.core.exceptions import MLXUserError

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required for one-class image datasets.") from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_one_class_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        required_paths=("train/normal", "val/normal"),
        dataset_label="one-class image dataset",
    )


def build_image_transform(
    *,
    height: int,
    width: int,
    colored: bool,
    augment: bool = False,
):
    if height < 1 or width < 1:
        raise MLXUserError("--height and --width must be positive.")
    operations = [transforms.Resize((height, width))]
    if augment:
        operations.extend((transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)))
    operations.append(transforms.ToTensor())
    operations.append(
        transforms.Normalize(
            (0.485, 0.456, 0.406) if colored else (0.5,),
            (0.229, 0.224, 0.225) if colored else (0.5,),
        )
    )
    return transforms.Compose(operations)


def load_image_tensor(
    path: str | Path,
    *,
    height: int,
    width: int,
    colored: bool,
) -> torch.Tensor:
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise MLXUserError(f"Input image not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            converted = image.convert("RGB" if colored else "L")
            return build_image_transform(
                height=height,
                width=width,
                colored=colored,
            )(converted)
    except (OSError, ValueError) as exc:
        raise MLXUserError(f"Cannot read input image '{image_path}': {exc}") from exc


class OneClassImageDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        height: int,
        width: int,
        colored: bool,
        normal_only: bool,
        augment: bool = False,
        transform=None,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser()
        self.split = split
        root = self.dataset_path / split
        self.root_dir = root if root.is_dir() else self.dataset_path
        if not self.root_dir.is_dir():
            raise MLXUserError(f"Dataset split not found: {self.root_dir}")
        if normal_only and _image_paths(self.root_dir / "anomaly"):
            raise MLXUserError(
                f"Anomalous images were found in the normal-only '{split}' split: "
                f"{self.root_dir / 'anomaly'}"
            )
        labels = (("normal", 0),) if normal_only else (("normal", 0), ("anomaly", 1))
        self.samples = [
            (path, label)
            for label_name, label in labels
            for path in _image_paths(self.root_dir / label_name)
        ]
        if not self.samples:
            expected = "normal images" if normal_only else "normal and anomaly images"
            raise MLXUserError(f"No {expected} were found under: {self.root_dir}")
        if not normal_only:
            present = {label for _, label in self.samples}
            if present != {0, 1}:
                raise MLXUserError(
                    f"Benchmark split must contain both normal and anomaly images: {self.root_dir}"
                )
        self.colored = colored
        self.transform = transform or build_image_transform(
            height=height,
            width=width,
            colored=colored,
            augment=augment,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB" if self.colored else "L"))
        except (OSError, ValueError) as exc:
            raise MLXUserError(f"Cannot read dataset image '{path}': {exc}") from exc
        return tensor, torch.tensor(label, dtype=torch.long), str(path)


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


__all__ = [
    "IMAGE_EXTENSIONS",
    "OneClassImageDataset",
    "build_image_transform",
    "image_one_class_dataset_root",
    "load_image_tensor",
]
