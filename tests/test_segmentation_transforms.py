from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.data import (
    SegmentationDataset,
    evaluation_segmentation_transform,
    transform_segmentation_pair,
)
from mlx.modes.segmentation.utils import checkpoint_payload


def _aligned_pair(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (np.arange(height * width).reshape(height, width) % 9).astype(np.uint8)
    image = np.repeat((mask * 20)[..., None], 3, axis=2)
    return image, mask


def test_center_crop_keeps_image_and_mask_aligned() -> None:
    image, mask = _aligned_pair(6, 8)

    cropped_image, cropped_mask = transform_segmentation_pair(
        image,
        mask,
        input_size=(4, 4),
        transform="center-crop",
    )

    assert np.array_equal(cropped_mask, mask[1:5, 2:6])
    assert np.array_equal(cropped_image[..., 0], cropped_mask * 20)


def test_random_crop_is_seeded_and_keeps_pair_alignment() -> None:
    image, mask = _aligned_pair(7, 9)

    first = transform_segmentation_pair(
        image,
        mask,
        input_size=(4, 5),
        transform="random-crop",
        rng=random.Random(42),
    )
    second = transform_segmentation_pair(
        image,
        mask,
        input_size=(4, 5),
        transform="random-crop",
        rng=random.Random(42),
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[0][..., 0], first[1] * 20)


def test_crop_zero_pads_undersized_pairs_symmetrically() -> None:
    image = np.full((2, 3, 3), 120, dtype=np.uint8)
    mask = np.full((2, 3), 6, dtype=np.uint8)

    cropped_image, cropped_mask = transform_segmentation_pair(
        image,
        mask,
        input_size=(4, 4),
        transform="center-crop",
    )

    assert cropped_image.shape == (4, 4, 3)
    assert cropped_mask.shape == (4, 4)
    assert np.array_equal(cropped_mask[1:3, :3], mask)
    assert np.count_nonzero(cropped_mask == 6) == 6
    assert np.all(cropped_mask[[0, 3], :] == 0)


def test_random_crop_becomes_center_crop_for_evaluation() -> None:
    assert evaluation_segmentation_transform("random-crop") == "center-crop"
    assert evaluation_segmentation_transform("resize") == "resize"


def test_unknown_segmentation_transform_is_user_facing() -> None:
    image, mask = _aligned_pair(4, 4)
    with pytest.raises(MLXUserError, match="Unsupported segmentation transform"):
        transform_segmentation_pair(
            image,
            mask,
            input_size=(2, 2),
            transform="unknown",
        )


def test_cli_accepts_segmentation_transform_names() -> None:
    random_crop = build_parser().parse_args(["--transform", "random-crop"])
    center_crop = build_parser().parse_args(["--transform", "center-crop"])

    assert random_crop.transform == "random-crop"
    assert center_crop.transform == "center-crop"


def test_segmentation_checkpoint_records_transform() -> None:
    payload = checkpoint_payload(
        torch.nn.Conv2d(3, 2, kernel_size=1),
        model_name="unet",
        config={"num_classes": 2, "transform": "random-crop"},
    )

    assert payload["transform"] == "random-crop"


def test_validation_dataset_uses_deterministic_crop(tmp_path: Path) -> None:
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "masks").mkdir(parents=True)
        image, mask = _aligned_pair(6, 8)
        cv2.imwrite(str(tmp_path / split / "images/sample.png"), image)
        cv2.imwrite(str(tmp_path / split / "masks/sample.png"), mask)

    train = SegmentationDataset(
        tmp_path,
        split="train",
        input_size=(4, 4),
        num_classes=9,
        transform="random-crop",
    )
    validation = SegmentationDataset(
        tmp_path,
        split="val",
        input_size=(4, 4),
        num_classes=9,
        transform="random-crop",
    )

    assert train.transform == "random-crop"
    assert validation.transform == "center-crop"
    first_image, first_mask = validation[0]
    second_image, second_mask = validation[0]
    assert np.array_equal(first_image.numpy(), second_image.numpy())
    assert np.array_equal(first_mask.numpy(), second_mask.numpy())
