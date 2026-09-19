from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.datasets import resolve_split_dataset_root
from mlx.core.exceptions import MLXUserError
from mlx.modes.saliency_mapping.requests import BuildSaliencyDatasetRequest
from mlx.modes.segmentation.data import (
    IMAGE_EXTENSIONS,
    evaluation_segmentation_transform,
    normalize_segmentation_transform,
)


def saliency_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        required_paths=("train/images", "train/masks", "val/images", "val/masks"),
        dataset_label="saliency dataset",
    )


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def paired_samples(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    images = {path.stem: path for path in _image_paths(images_dir)}
    masks = {path.stem: path for path in _image_paths(masks_dir)}
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing saliency maps for stems: {', '.join(missing_masks[:5])}")
        if missing_images:
            details.append(f"missing images for stems: {', '.join(missing_images[:5])}")
        raise MLXUserError(f"Image/saliency-map mismatch: {'; '.join(details)}")
    return [(images[stem], masks[stem]) for stem in sorted(images)]


def transform_saliency_pair(
    image: np.ndarray,
    target: np.ndarray,
    *,
    input_size: tuple[int, int],
    transform: str,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] != target.shape[:2]:
        raise MLXUserError(
            "Saliency image and target dimensions must match before transformation: "
            f"image={image.shape[:2]}, target={target.shape[:2]}."
        )
    width, height = map(int, input_size)
    if min(width, height) < 1:
        raise MLXUserError("Saliency input width and height must be at least 1.")
    transform = normalize_segmentation_transform(transform)
    if transform == "resize":
        return (
            cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(target, (width, height), interpolation=cv2.INTER_LINEAR),
        )

    source_height, source_width = target.shape
    vertical = max(0, height - source_height)
    horizontal = max(0, width - source_width)
    border = (
        vertical // 2,
        vertical - vertical // 2,
        horizontal // 2,
        horizontal - horizontal // 2,
        cv2.BORDER_CONSTANT,
    )
    if vertical or horizontal:
        image = cv2.copyMakeBorder(image, *border, value=0)
        target = cv2.copyMakeBorder(target, *border, value=0)
    available_height, available_width = target.shape
    if transform == "random-crop":
        source = rng or random
        top = source.randint(0, available_height - height)
        left = source.randint(0, available_width - width)
    else:
        top = (available_height - height) // 2
        left = (available_width - width) // 2
    return (
        np.ascontiguousarray(image[top:top + height, left:left + width]),
        np.ascontiguousarray(target[top:top + height, left:left + width]),
    )


def load_saliency_image_tensor(
    image_path: str | Path,
    *,
    input_size: tuple[int, int],
    colored: bool = True,
    transform: str = "resize",
) -> torch.Tensor:
    image_path = Path(image_path)
    flag = cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise MLXUserError(f"Cannot read image: {image_path}")
    if colored:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    placeholder = np.zeros(image.shape[:2], dtype=np.float32)
    image, _ = transform_saliency_pair(
        image,
        placeholder,
        input_size=input_size,
        transform=evaluation_segmentation_transform(transform),
    )
    if not colored:
        image = image[..., None]
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0


def load_saliency_pair_tensors(
    image_path: Path,
    target_path: Path,
    *,
    input_size: tuple[int, int],
    colored: bool,
    transform: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    flag = cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), flag)
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise MLXUserError(f"Cannot read image: {image_path}")
    if target is None:
        raise MLXUserError(f"Cannot read saliency target: {target_path}")
    if colored:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, target = transform_saliency_pair(
        image,
        target,
        input_size=input_size,
        transform=transform,
    )
    if not colored:
        image = image[..., None]
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).float().div(255.0)
    target_tensor = torch.from_numpy(np.ascontiguousarray(target)).float().div(255.0).unsqueeze(0)
    return image_tensor, target_tensor.clamp_(0.0, 1.0)


class SaliencyDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        input_size: tuple[int, int],
        colored: bool = True,
        transform: str = "resize",
    ) -> None:
        root = Path(dataset_path).expanduser()
        split_path = root if (root / "images").is_dir() else root / split
        self.images_dir = split_path / "images"
        self.masks_dir = split_path / "masks"
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise MLXUserError(
                f"Saliency split '{split}' requires '{self.images_dir}' and '{self.masks_dir}'."
            )
        self.samples = paired_samples(self.images_dir, self.masks_dir)
        if not self.samples:
            raise MLXUserError(f"No paired saliency samples found under: {split_path}")
        self.input_size = tuple(input_size)
        self.colored = colored
        normalized = normalize_segmentation_transform(transform)
        self.transform = normalized if split == "train" else evaluation_segmentation_transform(normalized)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return load_saliency_pair_tensors(
            *self.samples[index],
            input_size=self.input_size,
            colored=self.colored,
            transform=self.transform,
        )


def load_saliency_datasets(
    dataset_path: str | Path,
    *,
    input_size: tuple[int, int],
    colored: bool,
    transform: str,
) -> tuple[SaliencyDataset, SaliencyDataset]:
    common = {
        "input_size": input_size,
        "colored": colored,
        "transform": transform,
    }
    return (
        SaliencyDataset(dataset_path, split="train", **common),
        SaliencyDataset(dataset_path, split="val", **common),
    )


class BuildSaliencyDataset:
    def __init__(
        self,
        request: BuildSaliencyDatasetRequest,
        *,
        reporter: WorkflowReporter | None = None,
        input_resolver: Callable[[BuildSaliencyDatasetRequest, int], BuildSaliencyDatasetRequest] | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.input_resolver = input_resolver

    def execute(self) -> dict[str, int | str]:
        source = Path(self.request.dataset_path).expanduser()
        images_dir, masks_dir = source / "images", source / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise MLXUserError(
                f"Expected saliency source directories '{images_dir}' and '{masks_dir}'."
            )
        samples = paired_samples(images_dir, masks_dir)
        request = self.input_resolver(self.request, len(samples)) if self.input_resolver else self.request
        counts = (request.train_count, request.val_count, request.test_count)
        if any(value is None for value in counts):
            raise MLXUserError(
                "Saliency dataset building requires --train-count, --val-count, and --test-count."
            )
        train_count, val_count, test_count = (int(value) for value in counts)
        if min(train_count, val_count, test_count) < 0:
            raise MLXUserError("Saliency split counts must be zero or greater.")
        if train_count + val_count + test_count > len(samples):
            raise MLXUserError("Requested saliency split counts exceed the available paired samples.")
        if not request.output_path:
            raise MLXUserError("Saliency dataset building requires --output.")
        output = Path(request.output_path).expanduser()
        if output.exists():
            if not request.overwrite:
                raise MLXUserError(f"Output directory '{output}' exists; pass --overwrite to replace it.")
            shutil.rmtree(output)
        random.Random(request.random_seed).shuffle(samples)
        boundaries = (train_count, train_count + val_count, train_count + val_count + test_count)
        groups = {
            "train": samples[:boundaries[0]],
            "val": samples[boundaries[0]:boundaries[1]],
            "test": samples[boundaries[1]:boundaries[2]],
        }
        for split, pairs in groups.items():
            for kind in ("images", "masks"):
                (output / split / kind).mkdir(parents=True, exist_ok=True)
            for image_path, mask_path in pairs:
                shutil.copy2(image_path, output / split / "images" / image_path.name)
                shutil.copy2(mask_path, output / split / "masks" / mask_path.name)
        result = {"output_path": str(output), **{name: len(rows) for name, rows in groups.items()}}
        emit(self.reporter, "success", f"Saliency dataset created at {output}", payload=result)
        return result


__all__ = [
    "BuildSaliencyDataset",
    "SaliencyDataset",
    "load_saliency_datasets",
    "load_saliency_image_tensor",
    "load_saliency_pair_tensors",
    "paired_samples",
    "saliency_dataset_root",
    "transform_saliency_pair",
]
