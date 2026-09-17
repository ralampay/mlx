from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mlx.core.datasets import resolve_split_dataset_root
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.requests import BuildSegmentationDatasetRequest

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SEGMENTATION_TRANSFORMS = ("resize", "random-crop", "center-crop")


def normalize_segmentation_transform(value: str | None) -> str:
    transform = str(value or "resize").strip().lower()
    if transform not in SEGMENTATION_TRANSFORMS:
        available = ", ".join(SEGMENTATION_TRANSFORMS)
        raise MLXUserError(
            f"Unsupported segmentation transform '{transform}'. "
            f"Available transforms: {available}."
        )
    return transform


def evaluation_segmentation_transform(value: str | None) -> str:
    transform = normalize_segmentation_transform(value)
    return "center-crop" if transform == "random-crop" else transform


def transform_segmentation_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    input_size: tuple[int, int],
    transform: str,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one spatial transform to an aligned image/mask pair."""

    if image.shape[:2] != mask.shape[:2]:
        raise MLXUserError(
            "Segmentation image and mask dimensions must match before transformation: "
            f"image={image.shape[:2]}, mask={mask.shape[:2]}."
        )
    width, height = (int(value) for value in input_size)
    if width < 1 or height < 1:
        raise MLXUserError("Segmentation input width and height must be at least 1.")

    transform = normalize_segmentation_transform(transform)
    if transform == "resize":
        return (
            cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST),
        )

    image, mask = _pad_segmentation_pair(image, mask, width=width, height=height)
    available_height, available_width = mask.shape[:2]
    if transform == "random-crop":
        random_source = rng or random
        top = random_source.randint(0, available_height - height)
        left = random_source.randint(0, available_width - width)
    else:
        top = (available_height - height) // 2
        left = (available_width - width) // 2
    return (
        np.ascontiguousarray(image[top : top + height, left : left + width]),
        np.ascontiguousarray(mask[top : top + height, left : left + width]),
    )


def _pad_segmentation_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_height, source_width = mask.shape[:2]
    horizontal = max(0, width - source_width)
    vertical = max(0, height - source_height)
    top = vertical // 2
    bottom = vertical - top
    left = horizontal // 2
    right = horizontal - left
    if not horizontal and not vertical:
        return image, mask
    border = (top, bottom, left, right, cv2.BORDER_CONSTANT)
    return (
        cv2.copyMakeBorder(image, *border, value=0),
        cv2.copyMakeBorder(mask, *border, value=0),
    )


def segmentation_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        required_paths=("train/images", "train/masks", "val/images", "val/masks"),
        dataset_label="segmentation dataset",
    )


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
    transform: str = "resize",
) -> torch.Tensor:
    flag = cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise MLXUserError(f"Cannot read image: {image_path}")

    if colored:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    placeholder_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    image, _ = transform_segmentation_pair(
        image,
        placeholder_mask,
        input_size=input_size,
        transform=evaluation_segmentation_transform(transform),
    )
    if not colored:
        image = image[..., None]
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


def load_segmentation_pair_tensors(
    image_path: Path,
    mask_path: Path,
    *,
    input_size: tuple[int, int],
    num_classes: int,
    colored: bool,
    transform: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_flag = cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), image_flag)
    if image is None:
        raise MLXUserError(f"Cannot read image: {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise MLXUserError(f"Cannot read mask: {mask_path}")

    if colored:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, mask = transform_segmentation_pair(
        image,
        mask,
        input_size=input_size,
        transform=transform,
    )
    if not colored:
        image = image[..., None]

    if num_classes <= 2:
        mask = (mask > 0).astype("int64")
    else:
        mask = mask.astype("int64")
        observed_values = {int(value) for value in np.unique(mask)}
        invalid_values = sorted(observed_values - set(range(num_classes)))
        if invalid_values:
            raise MLXUserError(
                f"Mask '{mask_path}' contains class ids outside 0..{num_classes - 1}: "
                f"{invalid_values}"
            )
    return (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0,
        torch.from_numpy(np.ascontiguousarray(mask)).long(),
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        input_size: tuple[int, int],
        num_classes: int,
        colored: bool = True,
        transform: str = "resize",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.input_size = input_size
        self.num_classes = num_classes
        self.colored = colored
        requested_transform = normalize_segmentation_transform(transform)
        self.transform = (
            requested_transform
            if split == "train"
            else evaluation_segmentation_transform(requested_transform)
        )

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
        return load_segmentation_pair_tensors(
            image_path,
            mask_path,
            input_size=self.input_size,
            num_classes=self.num_classes,
            colored=self.colored,
            transform=self.transform,
        )


class SegmentationEvaluationDataset(Dataset):
    def __init__(
        self,
        split_path: str | Path,
        *,
        input_size: tuple[int, int],
        num_classes: int,
        colored: bool = True,
        transform: str = "resize",
    ) -> None:
        self.split_path = Path(split_path)
        self.images_dir = self.split_path / "images"
        self.masks_dir = self.split_path / "masks"
        self.input_size = input_size
        self.num_classes = num_classes
        self.colored = colored
        self.transform = evaluation_segmentation_transform(transform)
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise MLXUserError(
                "Expected evaluation dataset structure:\n"
                f"{self.split_path}/images/<file>\n"
                f"{self.split_path}/masks/<file>"
            )
        self.samples = _paired_samples(self.images_dir, self.masks_dir)
        if not self.samples:
            raise MLXUserError(
                f"No paired image/mask samples were found under: {self.split_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        return load_segmentation_pair_tensors(
            image_path,
            mask_path,
            input_size=self.input_size,
            num_classes=self.num_classes,
            colored=self.colored,
            transform=self.transform,
        )


def resolve_segmentation_evaluation_split(
    dataset_path: str | Path,
    *,
    split: str,
) -> Path:
    dataset_root = Path(dataset_path).expanduser()
    if not dataset_root.exists():
        raise MLXUserError(f"Dataset path not found: {dataset_root}")
    if (dataset_root / "images").is_dir() and (dataset_root / "masks").is_dir():
        return dataset_root
    split_path = dataset_root / split
    if (split_path / "images").is_dir() and (split_path / "masks").is_dir():
        return split_path
    raise MLXUserError(
        f"Segmentation evaluation split '{split}' was not found under '{dataset_root}'. "
        f"Expected '{split_path}/images' and '{split_path}/masks', or pass a direct split directory."
    )


def resolve_optional_segmentation_test_split(
    dataset_path: str | Path,
) -> Path | None:
    """Return a valid optional test split, rejecting partially defined splits."""

    test_path = Path(dataset_path).expanduser() / "test"
    if not test_path.exists():
        return None
    if not test_path.is_dir():
        raise MLXUserError(
            f"Segmentation test partition must be a directory: {test_path}"
        )
    images_dir = test_path / "images"
    masks_dir = test_path / "masks"
    has_images = images_dir.is_dir()
    has_masks = masks_dir.is_dir()
    if not has_images or not has_masks:
        missing = [
            str(path)
            for path, exists in ((images_dir, has_images), (masks_dir, has_masks))
            if not exists
        ]
        raise MLXUserError(
            "Segmentation test partition is incomplete. "
            f"Expected both '{images_dir}' and '{masks_dir}'; missing {', '.join(missing)}."
        )

    samples = _paired_samples(images_dir, masks_dir)
    if not samples:
        raise MLXUserError(
            f"No paired image/mask samples were found under test partition: {test_path}"
        )
    return test_path


def load_segmentation_datasets(
    dataset_path: str | Path,
    *,
    input_size: tuple[int, int],
    num_classes: int,
    colored: bool = True,
    transform: str = "resize",
) -> tuple[SegmentationDataset, SegmentationDataset]:
    return (
        SegmentationDataset(
            dataset_path,
            split="train",
            input_size=input_size,
            num_classes=num_classes,
            colored=colored,
            transform=transform,
        ),
        SegmentationDataset(
            dataset_path,
            split="val",
            input_size=input_size,
            num_classes=num_classes,
            colored=colored,
            transform=transform,
        ),
    )


def iter_split_images(dataset_path: str | Path, split: str = "test") -> Iterable[Path]:
    images_dir = Path(dataset_path) / split / "images"
    if not images_dir.exists():
        raise MLXUserError(f"Dataset split images directory not found: {images_dir}")
    return _iter_image_paths(images_dir)


class BuildSegmentationDataset:
    def __init__(
        self,
        request: BuildSegmentationDatasetRequest,
        *,
        reporter: WorkflowReporter | None = None,
        input_resolver: Callable[
            [BuildSegmentationDatasetRequest, int],
            BuildSegmentationDatasetRequest,
        ]
        | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.input_resolver = input_resolver

    def execute(self) -> None:
        _build_segmentation_dataset(
            self.request,
            reporter=self.reporter,
            input_resolver=self.input_resolver,
        )


def build_segmentation_dataset(dataset_path: str) -> None:
    from mlx.modes.segmentation.presentation import (
        RichSegmentationReporter,
        resolve_segmentation_dataset_build_request,
    )

    return BuildSegmentationDataset(
        BuildSegmentationDatasetRequest(dataset_path=dataset_path),
        reporter=RichSegmentationReporter(),
        input_resolver=resolve_segmentation_dataset_build_request,
    ).execute()


def _build_segmentation_dataset(
    request: BuildSegmentationDatasetRequest,
    *,
    reporter: WorkflowReporter | None = None,
    input_resolver: Callable[
        [BuildSegmentationDatasetRequest, int],
        BuildSegmentationDatasetRequest,
    ]
    | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    dataset_path = request.dataset_path
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise MLXUserError(f"Dataset path not found: {dataset_root}")

    images_dir, masks_dir = _paired_source_directories(dataset_root)
    samples = _paired_samples(images_dir, masks_dir)
    if not samples:
        raise MLXUserError(f"No paired image/mask samples were found under: {dataset_root}")

    emit(
        reporter,
        "info",
        f"Found {len(samples)} paired segmentation samples.",
        payload={
            "event": "segmentation_dataset_summary",
            "images_dir": images_dir,
            "masks_dir": masks_dir,
            "pairs": len(samples),
        },
    )

    if input_resolver is not None:
        request = input_resolver(request, len(samples))

    train_count = request.train_count
    val_count = request.val_count
    test_count = request.test_count
    if train_count is None or val_count is None or test_count is None:
        raise MLXUserError(
            "Segmentation dataset building requires --train-count, --val-count, and --test-count."
        )
    train_count = int(train_count)
    val_count = int(val_count)
    test_count = int(test_count)
    if min(train_count, val_count, test_count) < 0:
        raise MLXUserError("Segmentation split counts must be zero or greater.")

    total_needed = train_count + val_count + test_count
    if len(samples) < total_needed:
        emit(
            reporter,
            "warning",
            f"Only {len(samples)} paired samples were found, less than requested total {total_needed}."
        )

    output_value = request.output_path
    if not output_value:
        raise MLXUserError(
            "Segmentation dataset building requires --output pointing to a destination directory."
        )
    output_path = Path(output_value)
    if output_path.exists():
        if not request.overwrite:
            raise MLXUserError(
                f"Output directory '{output_path}' already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "masks").mkdir(parents=True, exist_ok=True)

    random.Random(request.random_seed).shuffle(samples)
    splits = {
        "train": samples[:train_count],
        "val": samples[train_count : train_count + val_count],
        "test": samples[train_count + val_count : train_count + val_count + test_count],
    }

    emit(reporter, "info", "Splitting segmentation dataset...")
    for split, split_samples in splits.items():
        for image_path, mask_path in split_samples:
            shutil.copy2(image_path, output_path / split / "images" / image_path.name)
            shutil.copy2(mask_path, output_path / split / "masks" / mask_path.name)

    emit(reporter, "success", f"Segmentation dataset created successfully at {output_path}")
