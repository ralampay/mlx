from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import torch
from rich.table import Table
from torch.utils.data import Dataset

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import (
    confirm_action,
    console,
    print_info,
    print_success,
    print_warning,
    prompt_int,
    prompt_text,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_image_paths(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_tensor(
    image_path: Path,
    *,
    input_size: Tuple[int, int],
    colored: bool,
) -> torch.Tensor:
    if colored:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise MLXUserError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise MLXUserError(f"Cannot read image: {image_path}")
        image = image[..., None]

    image = cv2.resize(image, input_size)
    return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0


class OneShotPairDataset(Dataset):
    """Generates positive and negative image pairs for one-shot learning."""

    def __init__(
        self,
        root_dir: os.PathLike[str] | str,
        input_size: Tuple[int, int] = (105, 105),
        colored: bool = True,
        n_pairs_per_class: int = 100,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.input_size = input_size
        self.colored = colored
        self.n_pairs_per_class = n_pairs_per_class
        self.class_to_images = self._index_images()
        self.classes = list(self.class_to_images.keys())

    def _index_images(self) -> dict[int, List[Path]]:
        if not self.root_dir.exists():
            raise MLXUserError(f"Dataset directory not found: {self.root_dir}")

        class_to_images: dict[int, List[Path]] = {}
        for label, subdir in enumerate(sorted(self.root_dir.iterdir())):
            if not subdir.is_dir():
                continue
            image_files = _iter_image_paths(subdir)
            if len(image_files) >= 2:
                class_to_images[label] = image_files

        if not class_to_images:
            raise MLXUserError(
                f"No labels with at least two images were found under: {self.root_dir}"
            )

        return class_to_images

    def __len__(self) -> int:
        return len(self.classes) * self.n_pairs_per_class

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        same_class = random.random() < 0.5

        if same_class:
            class_label = random.choice(self.classes)
            first, second = random.sample(self.class_to_images[class_label], 2)
            label = 1.0
        else:
            first_class, second_class = random.sample(self.classes, 2)
            first = random.choice(self.class_to_images[first_class])
            second = random.choice(self.class_to_images[second_class])
            label = 0.0

        image_one = load_image_tensor(first, input_size=self.input_size, colored=self.colored)
        image_two = load_image_tensor(second, input_size=self.input_size, colored=self.colored)
        return image_one, image_two, torch.tensor(label, dtype=torch.float32)


def load_ic_one_shot_dataset(
    dataset_path: os.PathLike[str] | str,
    input_size: Tuple[int, int] = (105, 105),
    colored: bool = True,
    n_pairs_per_class: int = 100,
) -> tuple[OneShotPairDataset, OneShotPairDataset]:
    dataset_root = Path(dataset_path)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise MLXUserError(
            "Expected dataset structure:\n"
            f"{dataset_root}/train/<class_name>/img.png\n"
            f"{dataset_root}/val/<class_name>/img.png"
        )

    return (
        OneShotPairDataset(
            train_dir,
            input_size=input_size,
            colored=colored,
            n_pairs_per_class=n_pairs_per_class,
        ),
        OneShotPairDataset(
            val_dir,
            input_size=input_size,
            colored=colored,
            n_pairs_per_class=n_pairs_per_class,
        ),
    )


def _label_directories(dataset_path: Path) -> List[Path]:
    return sorted(path for path in dataset_path.iterdir() if path.is_dir())


def build_ic_one_shot(dataset_path: str) -> None:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise MLXUserError(f"Dataset path not found: {dataset_root}")

    label_dirs = _label_directories(dataset_root)
    print_info(f"Found {len(label_dirs)} label(s) under {dataset_root.name}")

    table = Table(title="Label Summary", show_lines=True)
    table.add_column("Label", style="cyan")
    table.add_column("Images", justify="right", style="magenta")

    label_counts: dict[str, int] = {}
    for label_dir in label_dirs:
        count = len(_iter_image_paths(label_dir))
        label_counts[label_dir.name] = count
        table.add_row(label_dir.name, str(count))

    console.print(table)

    train_count = prompt_int("How many images per label for TRAIN?")
    val_count = prompt_int("How many images per label for VAL?")
    test_count = prompt_int("How many images per label for TEST?")

    total_needed = train_count + val_count + test_count
    for label, count in label_counts.items():
        if count < total_needed:
            print_warning(
                f"Label '{label}' has only {count} images, less than requested total {total_needed}."
            )

    output_path = Path(prompt_text("Enter output path for split dataset"))
    if output_path.exists():
        confirm_action(f"Output directory '{output_path}' already exists. Overwrite?", abort=True)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (output_path / split).mkdir(exist_ok=True)

    print_info("Splitting dataset...")
    for label_dir in label_dirs:
        images = _iter_image_paths(label_dir)
        random.shuffle(images)
        splits = {
            "train": images[:train_count],
            "val": images[train_count : train_count + val_count],
            "test": images[
                train_count + val_count : train_count + val_count + test_count
            ],
        }

        for split, split_images in splits.items():
            out_dir = output_path / split / label_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, out_dir / image_path.name)

    print_success(f"Dataset created successfully at {output_path}")


def iter_dataset_images(dataset_path: os.PathLike[str] | str) -> Iterable[Path]:
    dataset_root = Path(dataset_path)
    for root, _, files in os.walk(dataset_root):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
