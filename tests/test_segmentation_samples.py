from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation import samples as segmentation_samples
from mlx.modes.segmentation import train as segmentation_train
from mlx.modes.segmentation.data import resolve_optional_segmentation_test_split
from mlx.modes.segmentation.samples import (
    GenerateSegmentationSamples,
    evenly_spaced_sample_indices,
)


class FixedBinaryModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        foreground = images[:, :1] * 8.0 - 4.0
        return torch.cat((-foreground, foreground), dim=1)


class FixedMulticlassModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        shape = (len(images), 1, images.shape[2], images.shape[3])
        return torch.cat(
            (
                torch.zeros(shape, device=images.device),
                torch.ones(shape, device=images.device),
                torch.full(shape, 2.0, device=images.device),
            ),
            dim=1,
        )


def _write_test_split(root: Path, count: int, *, grayscale: bool = False) -> Path:
    split = root / "test"
    (split / "images").mkdir(parents=True)
    (split / "masks").mkdir()
    for index in range(count):
        image = (
            np.zeros((8, 8), dtype=np.uint8)
            if grayscale
            else np.zeros((8, 8, 3), dtype=np.uint8)
        )
        image[2:6, 2:6] = 255
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        cv2.imwrite(str(split / "images" / f"sample_{index:02d}.png"), image)
        cv2.imwrite(str(split / "masks" / f"sample_{index:02d}.png"), mask)
    return split


def test_evenly_spaced_sample_indices_are_bounded_and_deterministic() -> None:
    assert evenly_spaced_sample_indices(4, 16) == [0, 1, 2, 3]
    selected = evenly_spaced_sample_indices(33, 16)
    assert len(selected) == 16
    assert len(set(selected)) == 16
    assert selected[0] == 0
    assert selected[-1] == 32
    assert selected == evenly_spaced_sample_indices(33, 16)


@pytest.mark.parametrize("colored", (True, False))
def test_generate_segmentation_samples_writes_individual_views_and_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colored: bool,
) -> None:
    split = _write_test_split(tmp_path / "dataset", 3, grayscale=not colored)
    checkpoint = tmp_path / "best-dice.pth"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "checkpoint_path": checkpoint,
        "class_names": ["background", "foreground"],
        "colored": colored,
        "input_size": (8, 8),
        "mask_threshold": 0.75,
        "model_name": "unet",
        "num_classes": 2,
        "palette": [[0, 0, 0], [255, 80, 80]],
    }
    monkeypatch.setattr(
        segmentation_samples,
        "load_checkpoint_bundle",
        lambda config: (FixedBinaryModel(), metadata),
    )

    output = tmp_path / "training"
    paths = GenerateSegmentationSamples(
        {
            "colored": colored,
            "device": "cpu",
            "mask_threshold": 1.0,
            "overlay_alpha": 0.4,
        },
        checkpoint_path=checkpoint,
        test_split_path=split,
        output_dir=output,
    ).execute()

    assert len(paths) == 3
    for directory in ("original", "ground_truth", "prediction", "overlay", "panels"):
        assert len(list((output / "samples" / directory).glob("*.png"))) == 3
    panel = cv2.imread(str(paths[0]))
    assert panel.shape == (36, 32, 3)
    prediction = cv2.imread(
        str(output / "samples" / "prediction" / "sample_00.png")
    )
    assert tuple(prediction[3, 3]) == (80, 80, 255)


def test_generate_segmentation_samples_uses_multiclass_argmax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = _write_test_split(tmp_path / "dataset", 1)
    checkpoint = tmp_path / "best-dice.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        segmentation_samples,
        "load_checkpoint_bundle",
        lambda config: (
            FixedMulticlassModel(),
            {
                "colored": True,
                "input_size": (8, 8),
                "mask_threshold": 0.5,
                "num_classes": 3,
                "palette": [[0, 0, 0], [255, 80, 80], [10, 20, 30]],
            },
        ),
    )
    output = tmp_path / "training"

    GenerateSegmentationSamples(
        {"device": "cpu"},
        checkpoint_path=checkpoint,
        test_split_path=split,
        output_dir=output,
    ).execute()

    prediction = cv2.imread(
        str(output / "samples" / "prediction" / "sample_00.png")
    )
    assert np.all(prediction == np.asarray([30, 20, 10], dtype=np.uint8))


def test_generate_samples_replaces_stale_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = _write_test_split(tmp_path / "dataset", 1)
    checkpoint = tmp_path / "best-dice.pth"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "colored": True,
        "input_size": (8, 8),
        "mask_threshold": 0.5,
        "num_classes": 2,
        "palette": [[0, 0, 0], [255, 80, 80]],
    }
    monkeypatch.setattr(
        segmentation_samples,
        "load_checkpoint_bundle",
        lambda config: (FixedBinaryModel(), metadata),
    )
    output = tmp_path / "training"
    stale = output / "samples" / "panels" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    GenerateSegmentationSamples(
        {"device": "cpu"},
        checkpoint_path=checkpoint,
        test_split_path=split,
        output_dir=output,
    ).execute()

    assert not stale.exists()
    assert (output / "samples" / "panels" / "sample_00.png").is_file()


def test_optional_test_split_is_skipped_or_rejected_when_malformed(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    assert resolve_optional_segmentation_test_split(dataset) is None

    (dataset / "test").mkdir(parents=True)
    with pytest.raises(MLXUserError, match="test partition is incomplete"):
        resolve_optional_segmentation_test_split(dataset)

    (dataset / "test" / "images").mkdir(parents=True)
    with pytest.raises(MLXUserError, match="test partition is incomplete"):
        resolve_optional_segmentation_test_split(dataset)

    (dataset / "test" / "masks").mkdir()
    with pytest.raises(MLXUserError, match="No paired image/mask samples"):
        resolve_optional_segmentation_test_split(dataset)

    cv2.imwrite(
        str(dataset / "test" / "images" / "unpaired.png"),
        np.zeros((4, 4, 3), dtype=np.uint8),
    )
    with pytest.raises(MLXUserError, match="missing masks"):
        resolve_optional_segmentation_test_split(dataset)


@pytest.mark.parametrize("model_registry", [None, object()])
def test_training_sample_generation_falls_back_to_best_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_registry,
) -> None:
    events = []
    command = segmentation_train.TrainSegmentationModel.__new__(
        segmentation_train.TrainSegmentationModel
    )
    command.config = {"device": "cpu"}
    command.model_registry = model_registry
    command.reporter = CallbackWorkflowReporter(events.append)
    command.paths = {
        "dice_checkpoint_path": tmp_path / "missing-best-dice.pth",
        "checkpoint_path": tmp_path / "best-loss.pth",
        "output_dir": tmp_path / "output",
    }
    command.paths["checkpoint_path"].write_bytes(b"checkpoint")
    captured = {}

    class FakeGenerateSamples:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)

        def execute(self):
            return []

    monkeypatch.setattr(
        segmentation_train,
        "GenerateSegmentationSamples",
        FakeGenerateSamples,
    )

    command._generate_test_samples(tmp_path / "dataset" / "test")

    assert captured["checkpoint_path"] == command.paths["checkpoint_path"]
    assert captured.get("model_registry") is model_registry
    assert any(event.level == "warning" for event in events)


def test_training_invokes_sample_generation_for_a_valid_test_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    test_split = _write_test_split(dataset_root, 2)
    training_dataset = TensorDataset(
        torch.zeros(2, 3, 8, 8),
        torch.zeros(2, 8, 8, dtype=torch.long),
    )
    monkeypatch.setattr(
        segmentation_train,
        "load_segmentation_datasets",
        lambda *args, **kwargs: (training_dataset, training_dataset),
    )
    monkeypatch.setattr(
        segmentation_train,
        "build_segmentation_model",
        lambda *args, **kwargs: nn.Conv2d(3, 2, kernel_size=1),
    )

    def fake_run_epochs(command, **kwargs):
        command.paths["dice_checkpoint_path"].write_bytes(b"best dice")

    monkeypatch.setattr(
        segmentation_train.TrainSegmentationModel,
        "_run_epochs",
        fake_run_epochs,
    )
    captured = {}

    class FakeGenerateSamples:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)

        def execute(self):
            return []

    monkeypatch.setattr(
        segmentation_train,
        "GenerateSegmentationSamples",
        FakeGenerateSamples,
    )
    output = tmp_path / "training"

    segmentation_train.TrainSegmentationModel(
        {
            "batch_size": 1,
            "class_names": "background,foreground",
            "colored": True,
            "dataset_path": str(dataset_root),
            "device": "cpu",
            "epochs": 1,
            "input_size": (8, 8),
            "lr": 1e-3,
            "model": "unet",
            "num_classes": 2,
            "output_path": str(output),
        }
    ).execute()

    assert captured["test_split_path"] == test_split
    assert captured["checkpoint_path"] == output / "unet.best-dice.pth"
    assert captured["output_dir"] == output
