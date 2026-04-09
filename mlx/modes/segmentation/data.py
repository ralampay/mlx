from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import torch
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
from rich.table import Table

import random
import shutil

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _iter_image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _paired_source_directories(dataset_root: Path) -> tuple[Path, Path]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise MLXUserError(
            "Expected segmentation source dataset structure:\n"
            f"{dataset_root}/images/<file>\n"
            f"{dataset_root}/masks/<file>"
        )
    return images_dir, masks_dir


def _paired_samples(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = _iter_image_paths(images_dir)
    mask_paths = _iter_image_paths(masks_dir)
    image_map = {path.stem: path for path in image_paths}
    mask_map = {path.stem: path for path in mask_paths}

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    if missing_masks or missing_images:
        problems = []
        if missing_masks:
            problems.append(f"missing masks for stems: {', '.join(missing_masks[:5])}")
        if missing_images:
            problems.append(f"missing images for stems: {', '.join(missing_images[:5])}")
        raise MLXUserError(f"Image/mask mismatch: {'; '.join(problems)}")

    return [(image_map[stem], mask_map[stem]) for stem in sorted(image_map)]


def load_image_tensor(
    image_path: Path,
    *,
    input_size: tuple[int, int],
    colored: bool,
) -> torch.Tensor:
    flag = cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise MLXUserError(f"Cannot read image: {image_path}")

    if colored:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image[..., None]

    image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0


def load_mask_tensor(
    mask_path: Path,
    *,
    input_size: tuple[int, int],
    num_classes: int,
) -> torch.Tensor:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise MLXUserError(f"Cannot read mask: {mask_path}")

    mask = cv2.resize(mask, input_size, interpolation=cv2.INTER_NEAREST)
    if num_classes <= 2:
        mask = (mask > 0).astype("int64")
    else:
        mask = mask.astype("int64")
        valid_values = set(range(num_classes))
        observed_values = {int(value) for value in torch.from_numpy(mask).unique().tolist()}
        invalid_values = sorted(observed_values - valid_values)
        if invalid_values:
            raise MLXUserError(
                f"Mask '{mask_path}' contains class ids outside 0..{num_classes - 1}: {invalid_values}"
            )
    return torch.from_numpy(mask).long()


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        input_size: tuple[int, int],
        num_classes: int,
        colored: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.input_size = input_size
        self.num_classes = num_classes
        self.colored = colored

        split_dir = self.dataset_path / split
        self.images_dir = split_dir / "images"
        self.masks_dir = split_dir / "masks"
        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise MLXUserError(
                "Expected dataset structure:\n"
                f"{self.dataset_path}/train/images/<file>\n"
                f"{self.dataset_path}/train/masks/<file>\n"
                f"{self.dataset_path}/val/images/<file>\n"
                f"{self.dataset_path}/val/masks/<file>"
            )

        self.samples = _paired_samples(self.images_dir, self.masks_dir)
        if not self.samples:
            raise MLXUserError(f"No paired image/mask samples were found under: {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = load_image_tensor(image_path, input_size=self.input_size, colored=self.colored)
        mask = load_mask_tensor(mask_path, input_size=self.input_size, num_classes=self.num_classes)
        return image, mask


def load_segmentation_datasets(
    dataset_path: str | Path,
    *,
    input_size: tuple[int, int],
    num_classes: int,
    colored: bool = True,
) -> tuple[SegmentationDataset, SegmentationDataset]:
    return (
        SegmentationDataset(
            dataset_path,
            split="train",
            input_size=input_size,
            num_classes=num_classes,
            colored=colored,
        ),
        SegmentationDataset(
            dataset_path,
            split="val",
            input_size=input_size,
            num_classes=num_classes,
            colored=colored,
        ),
    )


def iter_split_images(dataset_path: str | Path, split: str = "test") -> Iterable[Path]:
    images_dir = Path(dataset_path) / split / "images"
    if not images_dir.exists():
        raise MLXUserError(f"Dataset split images directory not found: {images_dir}")
    return _iter_image_paths(images_dir)


def build_segmentation_dataset(dataset_path: str) -> None:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise MLXUserError(f"Dataset path not found: {dataset_root}")

    images_dir, masks_dir = _paired_source_directories(dataset_root)
    samples = _paired_samples(images_dir, masks_dir)
    if not samples:
        raise MLXUserError(f"No paired image/mask samples were found under: {dataset_root}")

    table = Table(title="Segmentation Pair Summary", show_lines=True)
    table.add_column("Directory", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Images Dir", str(images_dir))
    table.add_row("Masks Dir", str(masks_dir))
    table.add_row("Pairs", str(len(samples)))
    console.print(table)

    train_count = prompt_int("How many paired samples for TRAIN?")
    val_count = prompt_int("How many paired samples for VAL?")
    test_count = prompt_int("How many paired samples for TEST?")

    total_needed = train_count + val_count + test_count
    if len(samples) < total_needed:
        print_warning(
            f"Only {len(samples)} paired samples were found, less than requested total {total_needed}."
        )

    output_path = Path(prompt_text("Enter output path for split dataset"))
    if output_path.exists():
        confirm_action(f"Output directory '{output_path}' already exists. Overwrite?", abort=True)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "masks").mkdir(parents=True, exist_ok=True)

    random.shuffle(samples)
    splits = {
        "train": samples[:train_count],
        "val": samples[train_count : train_count + val_count],
        "test": samples[train_count + val_count : train_count + val_count + test_count],
    }

    print_info("Splitting segmentation dataset...")
    for split, split_samples in splits.items():
        for image_path, mask_path in split_samples:
            shutil.copy2(image_path, output_path / split / "images" / image_path.name)
            shutil.copy2(mask_path, output_path / split / "masks" / mask_path.name)

    print_success(f"Segmentation dataset created successfully at {output_path}")
