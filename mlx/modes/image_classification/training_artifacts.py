"""Local classification history and checkpoint persistence."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.utils import (
    load_training_checkpoint, save_checkpoint, save_training_checkpoint, update_checkpoint_model,
)

TRAINING_CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
]
SVDD_TRAINING_CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "train_classification_loss",
    "train_svdd_loss",
    "val_loss",
    "val_classification_loss",
    "val_svdd_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def _write_training_csv(
    csv_path: Path,
    history: list[dict[str, float | int]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        columns = (
            SVDD_TRAINING_CSV_COLUMNS
            if history and "train_svdd_loss" in history[0]
            else TRAINING_CSV_COLUMNS
        )
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    **{
                        column: f"{float(row[column]):.6f}"
                        for column in columns[1:]
                    },
                }
            )


class ClassificationTrainingArtifacts:
    """Filesystem backend; callers may inject another object with these operations."""

    write_history = staticmethod(_write_training_csv)
    save_model = staticmethod(save_checkpoint)
    save_training = staticmethod(save_training_checkpoint)
    load_training = staticmethod(load_training_checkpoint)
    update_model = staticmethod(update_checkpoint_model)

    @staticmethod
    def load_model(path, model, device):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        return checkpoint

    def restore(
        self,
        config: dict[str, Any],
        model,
        optimizer,
        *,
        model_name: str,
        family: str,
        checkpoint_path: Path,
        training_csv_path: Path,
        classes: list[str] | None = None,
        reporter: WorkflowReporter | None = None,
    ) -> tuple[int, float, list[dict[str, float | int]]]:
        reporter = reporter or NullWorkflowReporter()
        resume_path = config.get("model_path")
        if not resume_path:
            self.write_history(training_csv_path, [])
            return 0, float("inf"), []

        training_state = self.load_training(
            resume_path,
            model,
            optimizer,
            model_name=model_name,
            family=family,
            config=config,
            classes=classes,
        )
        completed_epoch = training_state["completed_epoch"]
        target_epochs = int(config.get("epochs", 0))
        if completed_epoch > target_epochs:
            raise MLXUserError(
                f"Resume checkpoint has completed epoch {completed_epoch}, which exceeds "
                f"the requested --epochs target of {target_epochs}."
            )
        history = training_state["history"]
        self.write_history(training_csv_path, history)
        if not checkpoint_path.exists():
            self.save_model(
                checkpoint_path,
                model,
                model_name=model_name,
                family=family,
                config=config,
                classes=classes,
            )
        emit(
            reporter,
            "info",
            f"Resuming {model_name} from completed epoch {completed_epoch}: {resume_path}"
        )
        return completed_epoch, training_state["best_val_loss"], history
