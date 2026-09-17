from __future__ import annotations

import torch
from torch import nn

from mlx.core.commands import CallbackWorkflowReporter
from mlx.modes.segmentation import train


def test_segmentation_epoch_event_exposes_standard_training_metrics(
    tmp_path, monkeypatch
) -> None:
    events = []
    command = train.TrainSegmentationModel(
        {
            "batch_size": 1,
            "class_names": "background,foreground",
            "colored": True,
            "dataset_path": str(tmp_path / "dataset"),
            "device": "cpu",
            "epochs": 2,
            "input_size": (8, 8),
            "lr": 0.001,
            "model": "unet",
            "num_classes": 2,
            "output_path": str(tmp_path / "output"),
        },
        reporter=CallbackWorkflowReporter(events.append),
    )
    model = nn.Conv2d(3, 2, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = iter((1.0, 0.8))
    validation = iter(
        (
            (
                0.9,
                {
                    "pixel_accuracy": 0.5,
                    "macro_dice": 0.4,
                    "mean_foreground_dice": 0.3,
                    "mean_iou": 0.25,
                    "mean_foreground_iou": 0.2,
                },
                [],
            ),
            (
                0.7,
                {
                    "pixel_accuracy": 0.6,
                    "macro_dice": 0.5,
                    "mean_foreground_dice": 0.4,
                    "mean_iou": 0.35,
                    "mean_foreground_iou": 0.3,
                },
                [],
            ),
        )
    )
    monkeypatch.setattr(command, "_train_epoch", lambda *args: next(train_losses))
    monkeypatch.setattr(command, "_validate", lambda *args: next(validation))
    monkeypatch.setattr(train, "write_csv", lambda *args: None)
    monkeypatch.setattr(train, "write_training_curves", lambda *args: None)
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train, "save_training_checkpoint", lambda *args, **kwargs: None
    )

    command._run_epochs(
        model=model,
        train_loader=None,
        val_loader=None,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        start_epoch=0,
        best_val_loss=float("inf"),
        best_dice=float("-inf"),
        history=[],
    )

    epoch_events = [
        event
        for event in events
        if isinstance(event.payload, dict)
        and event.payload.get("event") == "segmentation_epoch"
    ]
    assert len(epoch_events) == 2
    first, second = (event.payload for event in epoch_events)
    assert first["metrics"]["train_loss"] == 1.0
    assert first["metrics"]["val_loss"] == 0.9
    assert first["metrics"]["learning_rate"] == 0.001
    assert first["metrics"]["epoch_seconds"] >= 0
    assert first["is_best_val_loss"] is True
    assert first["is_best_foreground_dice"] is True
    assert second["previous_metrics"]["epoch"] == 1
    assert second["previous_metrics"]["val_loss"] == 0.9
    assert second["is_best_val_loss"] is True
    assert second["is_best_foreground_dice"] is True
    assert len(second["checkpoints"]) == 3
