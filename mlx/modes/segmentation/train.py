from __future__ import annotations

import platform
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.data import load_segmentation_datasets
from mlx.modes.segmentation.metrics import (
    aggregate_confusion_metrics,
    class_metrics_from_confusion,
    confusion_matrix_from_arrays,
    metric_slug,
)
from mlx.modes.segmentation.models import build_segmentation_model
from mlx.modes.segmentation.research import (
    write_csv,
    write_json,
    write_training_curves,
)
from mlx.modes.segmentation.requests import SegmentationRequest
from mlx.modes.segmentation.utils import (
    load_training_checkpoint,
    resolve_class_names,
    resolve_model_name,
    resolve_train_output_paths,
    save_checkpoint,
    save_training_checkpoint,
)


class TrainSegmentationModel:
    def __init__(
        self,
        config: dict[str, Any] | SegmentationRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        if isinstance(config, SegmentationRequest):
            config = config.to_config()
        self.config = dict(config)
        self.model_name = resolve_model_name(config)
        self.device = str(config["device"])
        self.batch_size = max(1, int(config.get("batch_size", 4)))
        self.epochs = int(config.get("epochs", 50))
        self.learning_rate = float(config.get("lr") or 1e-3)
        self.input_size = tuple(config.get("input_size", (256, 256)))
        self.num_classes = int(config.get("num_classes", 2))
        self.colored = bool(config.get("colored", True))
        self.class_names = resolve_class_names(config, self.num_classes)
        self.config["class_names"] = self.class_names
        self.paths = resolve_train_output_paths(config, model_name=self.model_name)
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> None:
        if self.epochs < 1:
            raise MLXUserError("--epochs must be at least 1 for segmentation training.")
        if self.num_classes < 2:
            raise MLXUserError("--num-classes must be at least 2 for segmentation training.")
        emit(
            self.reporter,
            "info",
            f"Starting segmentation training on device={self.device} for {self.epochs} epochs"
        )
        train_dataset, val_dataset = load_segmentation_datasets(
            self.config["dataset_path"],
            input_size=self.input_size,
            num_classes=self.num_classes,
            colored=self.colored,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )
        model = build_segmentation_model(
            self.model_name,
            self.config,
            num_classes=self.num_classes,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        start_epoch, best_val_loss, best_dice, history = self._prepare_state(
            model,
            optimizer,
        )
        if start_epoch > self.epochs:
            raise MLXUserError(
                f"Resume checkpoint completed epoch {start_epoch}, which exceeds --epochs={self.epochs}."
            )

        self.paths["output_dir"].mkdir(parents=True, exist_ok=True)
        write_json(
            self.paths["training_config_path"],
            {
                "config": self.config,
                "model_name": self.model_name,
                "class_names": self.class_names,
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
            },
        )
        self._run_epochs(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            best_dice=best_dice,
            history=history,
        )
        emit(
            self.reporter,
            "success",
            f"Segmentation training complete; research artifacts are in {self.paths['output_dir']}"
        )

    def _run_epochs(
        self,
        *,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        start_epoch: int,
        best_val_loss: float,
        best_dice: float,
        history: list[dict[str, Any]],
    ) -> None:
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.perf_counter()
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_metrics, class_rows = self._validate(model, val_loader, criterion)
            row = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            for class_row in class_rows:
                slug = metric_slug(str(class_row["class_name"]))
                for metric in ("precision", "recall", "specificity", "dice", "iou"):
                    row[f"{slug}_{metric}"] = class_row[metric]
            history.append(row)
            write_csv(self.paths["training_csv_path"], history)
            write_training_curves(self.paths["training_curves_path"], history)

            messages: list[str] = []
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.paths["checkpoint_path"], model, model_name=self.model_name, config=self.config
                )
                messages.append(f"best loss → {self.paths['checkpoint_path']}")
            foreground_dice = float(val_metrics["mean_foreground_dice"])
            if np.isfinite(foreground_dice) and foreground_dice > best_dice:
                best_dice = foreground_dice
                save_checkpoint(
                    self.paths["dice_checkpoint_path"], model, model_name=self.model_name, config=self.config
                )
                messages.append(f"best Dice → {self.paths['dice_checkpoint_path']}")
            save_training_checkpoint(
                self.paths["last_checkpoint_path"],
                model,
                optimizer,
                model_name=self.model_name,
                config=self.config,
                completed_epoch=epoch + 1,
                best_val_loss=best_val_loss,
                best_foreground_dice=best_dice,
                history=history,
            )
            messages.append(f"last → {self.paths['last_checkpoint_path']}")
            emit(
                self.reporter,
                "progress",
                f"Completed segmentation epoch {epoch + 1}/{self.epochs}.",
                current=epoch + 1,
                total=self.epochs,
                payload={
                    "event": "segmentation_epoch",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "metrics": val_metrics,
                    "checkpoints": messages,
                },
            )

    def _prepare_state(self, model, optimizer) -> tuple[int, float, float, list[dict[str, Any]]]:
        resume_path = self.config.get("model_path")
        if not resume_path:
            write_csv(self.paths["training_csv_path"], [])
            return 0, float("inf"), float("-inf"), []
        state = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            model_name=self.model_name,
            config=self.config,
        )
        history = list(state["history"])
        write_csv(self.paths["training_csv_path"], history)
        if not self.paths["checkpoint_path"].exists():
            save_checkpoint(
                self.paths["checkpoint_path"],
                model,
                model_name=self.model_name,
                config=self.config,
            )
        return (
            int(state["completed_epoch"]),
            float(state["best_val_loss"]),
            float(state["best_foreground_dice"]),
            history,
        )

    def _train_epoch(
        self,
        model,
        loader,
        criterion,
        optimizer,
    ) -> float:
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(images)
            sample_count += len(images)
        return running_loss / max(1, sample_count)

    def _validate(self, model, loader, criterion) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
        model.eval()
        loss_sum = 0.0
        sample_count = 0
        targets: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for images, masks in loader:
                images, masks_device = images.to(self.device), masks.to(self.device)
                logits = model(images)
                loss_sum += float(criterion(logits, masks_device).item()) * len(images)
                sample_count += len(images)
                targets.append(masks.numpy())
                predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets_np = np.concatenate(targets, axis=0)
        predictions_np = np.concatenate(predictions, axis=0)
        matrix = confusion_matrix_from_arrays(
            targets_np,
            predictions_np,
            self.num_classes,
        )
        class_rows = class_metrics_from_confusion(matrix, self.class_names)
        metrics = aggregate_confusion_metrics(matrix, class_rows)
        return loss_sum / max(1, sample_count), metrics, class_rows


def train_segmentation(config: dict[str, Any]) -> None:
    from mlx.modes.segmentation.presentation import RichSegmentationReporter

    TrainSegmentationModel(config, reporter=RichSegmentationReporter()).execute()


class SmokeTestSegmentationModel:
    def __init__(
        self,
        request: SegmentationRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> None:
        _run_smoke_test(self.request.to_config(), reporter=self.reporter)


def smoke_test_segmentation(config: dict[str, Any]) -> None:
    from mlx.modes.segmentation.presentation import RichSegmentationReporter

    SmokeTestSegmentationModel(
        SegmentationRequest.from_config(config),
        reporter=RichSegmentationReporter(),
    ).execute()


def _run_smoke_test(
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    model_name = resolve_model_name(config)
    batch = int(config["batch_size"])
    width, height = config["input_size"]
    device = config["device"]
    num_classes = int(config.get("num_classes", 2))
    emit(
        reporter,
        "info",
        f"Running segmentation test on device={device} | input={width}x{height} "
        f"| batch={batch} | classes={num_classes}"
    )
    model = build_segmentation_model(model_name, config, num_classes=num_classes).to(device)
    channels = 3 if config.get("colored", True) else 1
    output = model(torch.randn(batch, channels, height, width).to(device))
    emit(
        reporter,
        "success",
        "Test completed successfully!",
        payload={
            "event": "segmentation_tensor_output",
            "shape": list(output.shape),
            "values": output.flatten().tolist()[:16],
        },
    )
