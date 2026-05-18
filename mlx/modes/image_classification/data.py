from __future__ import annotations

import os
import random
import shutil
from math import floor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import torch
from rich.table import Table
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - pillow is expected via torchvision
    raise ImportError(
        "Pillow is required for image-classification datasets. Install it with 'pip install pillow'."
    ) from exc

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import (
    confirm_action,
    console,
    print_info,
    print_success,
    print_warning,
    prompt_float,
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


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_path: os.PathLike[str] | str,
        *,
        split: str = "train",
        transform=None,
        input_size: Tuple[int, int] = (224, 224),
        colored: bool = True,
        label_names: Sequence[str] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.input_size = input_size
        self.colored = colored
        self.transform = transform or _default_classification_transform(
            input_size=input_size,
            colored=colored,
        )
        candidate_split_dir = self.dataset_path / split

        if not self.dataset_path.exists():
            raise MLXUserError(f"Dataset directory not found: {self.dataset_path}")
        if candidate_split_dir.exists():
            self.root_dir = candidate_split_dir
        else:
            self.root_dir = self.dataset_path
        if not self.root_dir.exists():
            raise MLXUserError(f"Dataset directory not found: {self.root_dir}")

        discovered_dirs = {path.name: path for path in _label_directories(self.root_dir)}
        if label_names is None:
            self.label_names = sorted(discovered_dirs.keys())
        else:
            self.label_names = list(label_names)

        self.label_to_index = {label: index for index, label in enumerate(self.label_names)}
        self.samples: list[tuple[Path, int]] = []

        for label in self.label_names:
            label_dir = discovered_dirs.get(label)
            if label_dir is None:
                continue
            for image_path in _iter_image_paths(label_dir):
                self.samples.append((image_path, self.label_to_index[label]))

        if not self.samples:
            raise MLXUserError(f"No labelled images were found under: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path)
        if self.colored:
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def _default_classification_transform(
    *,
    input_size: Tuple[int, int],
    colored: bool,
    is_training: bool = False,
):
    transform_steps = [transforms.Resize(input_size)]
    if is_training:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    if not colored:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    transform_steps.append(transforms.ToTensor())
    if colored:
        transform_steps.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    else:
        transform_steps.append(
            transforms.Normalize(
                mean=(0.5,),
                std=(0.5,),
            )
        )
    return transforms.Compose(transform_steps)


def load_one_shot_datasets(
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


def load_standard_classification_datasets(
    dataset_path: os.PathLike[str] | str,
    *,
    input_size: Tuple[int, int] = (224, 224),
    colored: bool = True,
    apply_transformations: bool = False,
) -> tuple[ImageClassificationDataset, ImageClassificationDataset, list[str]]:
    dataset_root = Path(dataset_path)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise MLXUserError(
            "Expected dataset structure:\n"
            f"{dataset_root}/train/<class_name>/img.png\n"
            f"{dataset_root}/val/<class_name>/img.png"
        )

    label_names = [path.name for path in _label_directories(train_dir)]
    if not label_names:
        raise MLXUserError(f"No label directories were found under: {train_dir}")

    train_transform = _default_classification_transform(
        input_size=input_size,
        colored=colored,
        is_training=apply_transformations,
    )
    val_transform = _default_classification_transform(
        input_size=input_size,
        colored=colored,
        is_training=False,
    )
    train_dataset = ImageClassificationDataset(
        dataset_root,
        split="train",
        transform=train_transform,
        input_size=input_size,
        colored=colored,
        label_names=label_names,
    )
    val_dataset = ImageClassificationDataset(
        dataset_root,
        split="val",
        transform=val_transform,
        input_size=input_size,
        colored=colored,
        label_names=label_names,
    )
    return train_dataset, val_dataset, label_names


def load_standard_classification_directory(
    dataset_path: os.PathLike[str] | str,
    *,
    label_names: Sequence[str],
    split: str = "test",
    transform=None,
    input_size: Tuple[int, int] = (224, 224),
    colored: bool = True,
) -> ImageClassificationDataset:
    return ImageClassificationDataset(
        dataset_path,
        split=split,
        transform=transform,
        input_size=input_size,
        colored=colored,
        label_names=label_names,
    )


def _label_directories(dataset_path: Path) -> List[Path]:
    return sorted(path for path in dataset_path.iterdir() if path.is_dir())


def _resolve_split_count(value: int | None, prompt: str) -> int:
    return prompt_int(prompt) if value is None else value


def _resolve_split_ratio(value: float | None, prompt: str) -> float:
    return prompt_float(prompt) if value is None else value


def _resolve_split_mode(
    split_mode: str | None,
    *,
    train_count: int | None,
    val_count: int | None,
    test_count: int | None,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
) -> str:
    has_counts = any(value is not None for value in (train_count, val_count, test_count))
    has_ratios = any(value is not None for value in (train_ratio, val_ratio, test_ratio))

    if has_counts and has_ratios:
        raise MLXUserError(
            "Split counts and split ratios cannot be used together. Choose one mode."
        )

    resolved_mode = split_mode
    if resolved_mode is None:
        resolved_mode = "ratios" if has_ratios else "counts"

    if resolved_mode == "counts" and has_ratios:
        raise MLXUserError("Split mode 'counts' cannot be used with ratio arguments.")
    if resolved_mode == "ratios" and has_counts:
        raise MLXUserError("Split mode 'ratios' cannot be used with count arguments.")

    return resolved_mode


def _counts_from_ratios(
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    label_count: int,
) -> tuple[int, int, int]:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise MLXUserError("Split ratios must be zero or greater.")

    ratio_total = sum(ratios)
    if ratio_total <= 0:
        raise MLXUserError("At least one split ratio must be greater than zero.")

    normalized = [ratio / ratio_total for ratio in ratios]
    raw_counts = [label_count * ratio for ratio in normalized]
    counts = [floor(value) for value in raw_counts]
    remainder = label_count - sum(counts)

    ranked_indices = sorted(
        range(len(raw_counts)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in ranked_indices[:remainder]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def build_image_classification_dataset(
    dataset_path: str,
    *,
    train_count: int | None = None,
    val_count: int | None = None,
    test_count: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    split_mode: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    random_seed: int | None = None,
) -> None:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise MLXUserError(f"Dataset path not found: {dataset_root}")

    label_dirs = _label_directories(dataset_root)
    if not label_dirs:
        raise MLXUserError(f"No label directories were found under: {dataset_root}")

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

    resolved_split_mode = _resolve_split_mode(
        split_mode,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if resolved_split_mode == "ratios":
        train_ratio = _resolve_split_ratio(train_ratio, "Train ratio?")
        val_ratio = _resolve_split_ratio(val_ratio, "Validation ratio?")
        test_ratio = _resolve_split_ratio(test_ratio, "Test ratio?")
        print_info(
            "Ratio mode splits each label independently using normalized ratios. "
            f"Ratios -> train: {train_ratio}, val: {val_ratio}, test: {test_ratio}."
        )
    else:
        train_count = _resolve_split_count(train_count, "How many images per label for TRAIN?")
        val_count = _resolve_split_count(val_count, "How many images per label for VAL?")
        test_count = _resolve_split_count(test_count, "How many images per label for TEST?")

    if resolved_split_mode == "counts":
        if train_count < 0 or val_count < 0 or test_count < 0:
            raise MLXUserError("Split counts must be zero or greater.")
        total_needed = train_count + val_count + test_count
        for label, count in label_counts.items():
            if count < total_needed:
                print_warning(
                    f"Label '{label}' has only {count} images, less than requested total {total_needed}."
                )

    interactive_output = output_path is None
    resolved_output_path = (
        Path(prompt_text("Enter output path for split dataset"))
        if interactive_output
        else Path(output_path)
    )
    if resolved_output_path.exists():
        if interactive_output and not overwrite:
            confirm_action(
                f"Output directory '{resolved_output_path}' already exists. Overwrite?",
                abort=True,
            )
        elif not overwrite:
            raise MLXUserError(
                f"Output directory '{resolved_output_path}' already exists. "
                "Re-run with --overwrite to replace it."
            )
        shutil.rmtree(resolved_output_path)
    resolved_output_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (resolved_output_path / split).mkdir(exist_ok=True)

    rng = random.Random(random_seed)

    print_info("Splitting dataset...")
    for label_dir in label_dirs:
        images = _iter_image_paths(label_dir)
        rng.shuffle(images)
        if resolved_split_mode == "ratios":
            train_count, val_count, test_count = _counts_from_ratios(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                label_count=len(images),
            )
        splits = {
            "train": images[:train_count],
            "val": images[train_count : train_count + val_count],
            "test": images[
                train_count + val_count : train_count + val_count + test_count
            ],
        }

        for split, split_images in splits.items():
            out_dir = resolved_output_path / split / label_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, out_dir / image_path.name)

    print_success(f"Dataset created successfully at {resolved_output_path}")


def resolve_evaluation_dir(dataset_path: os.PathLike[str] | str) -> Path:
    dataset_root = Path(dataset_path)
    test_dir = dataset_root / "test"
    return test_dir if test_dir.exists() else dataset_root


def iter_dataset_images(dataset_path: os.PathLike[str] | str) -> Iterable[Path]:
    dataset_root = Path(dataset_path)
    for root, _, files in os.walk(dataset_root):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
