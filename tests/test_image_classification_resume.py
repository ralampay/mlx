import csv
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification import train
from mlx.modes.image_classification.utils import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))


def _config(tmp_path, **overrides):
    config = {
        "apply_transformations": False,
        "batch_size": 2,
        "colored": True,
        "dataset_path": str(tmp_path / "dataset"),
        "device": "cpu",
        "drax_fusion_mode": "average",
        "epochs": 1,
        "input_size": (4, 4),
        "lr": 0.001,
        "model": "resnet18",
        "model_path": None,
        "output_path": str(tmp_path / "artifacts"),
        "pretrained": False,
        "refresh_per_second": 2,
        "use_best": True,
        "verbose": False,
    }
    config.update(overrides)
    return config


def test_training_checkpoint_round_trip_restores_epoch_optimizer_and_rng(tmp_path):
    config = _config(tmp_path)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = model(torch.randn(2, 3, 4, 4)).sum()
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "resnet18.last.pth"
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_loss": 0.9,
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }
    ]

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_name="resnet18",
        family="standard",
        config=config,
        completed_epoch=1,
        best_val_loss=0.9,
        history=history,
        classes=["a", "b"],
    )
    expected_random = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    restored_model = _tiny_model()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
    state = load_training_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        model_name="resnet18",
        family="standard",
        config=config,
        classes=["a", "b"],
    )

    assert state["completed_epoch"] == 1
    assert state["best_val_loss"] == pytest.approx(0.9)
    assert state["history"] == history
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert random.random() == pytest.approx(expected_random)
    assert np.random.random() == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(1), expected_torch)
    for original, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(original, restored)


def test_resume_rejects_mismatched_classes(tmp_path):
    config = _config(tmp_path)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint_path = tmp_path / "resnet18.last.pth"
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_name="resnet18",
        family="standard",
        config=config,
        completed_epoch=0,
        best_val_loss=float("inf"),
        history=[],
        classes=["a", "b"],
    )

    with pytest.raises(MLXUserError, match="class labels"):
        load_training_checkpoint(
            checkpoint_path,
            _tiny_model(),
            torch.optim.Adam(_tiny_model().parameters()),
            model_name="resnet18",
            family="standard",
            config=config,
            classes=["a", "c"],
        )


def test_standard_training_resumes_without_repeating_epochs(monkeypatch, tmp_path):
    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(images, targets)
    monkeypatch.setattr(
        train,
        "load_standard_classification_datasets",
        lambda *args, **kwargs: (dataset, dataset, ["a", "b"]),
    )
    monkeypatch.setattr(
        train,
        "build_image_classification_model",
        lambda *args, **kwargs: _tiny_model(),
    )

    first_config = _config(tmp_path, epochs=1)
    train._train_standard("resnet18", first_config)
    last_checkpoint = tmp_path / "artifacts" / "resnet18.last.pth"
    second_config = _config(
        tmp_path,
        epochs=2,
        model_path=str(last_checkpoint),
    )
    train._train_standard("resnet18", second_config)

    with (tmp_path / "artifacts" / "training.csv").open(
        newline="", encoding="utf-8"
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["completed_epoch"] == 2
    assert len(checkpoint["history"]) == 2
