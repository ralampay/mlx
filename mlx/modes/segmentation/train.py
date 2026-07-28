from __future__ import annotations

import platform
import time
from typing import Any

import cv2
import numpy as np
import torch
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import nn, optim
from torch.utils.data import DataLoader

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_success
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
from mlx.modes.segmentation.utils import (
    load_training_checkpoint,
    resolve_class_names,
    resolve_model_name,
    resolve_train_output_paths,
    save_checkpoint,
    save_training_checkpoint,
)


class TrainSegmentationModel:
    def __init__(self, config: dict[str, Any]) -> None:
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

    def execute(self) -> None:
        if self.epochs < 1:
            raise MLXUserError("--epochs must be at least 1 for segmentation training.")
        if self.num_classes < 2:
            raise MLXUserError("--num-classes must be at least 2 for segmentation training.")
        print_info(
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
        progress = _build_progress(
            epochs=self.epochs,
            train_loader_size=len(train_loader),
        )
        epoch_task, batch_task = progress.task_ids
        progress.update(epoch_task, completed=start_epoch)
        checkpoint_panel = Panel(
            (
                f"Resumed at epoch {start_epoch} from {self.config['model_path']}"
                if start_epoch
                else "No model saved yet"
            ),
            title="Checkpoint",
            border_style="dim",
        )

        with Live(
            Group(progress, checkpoint_panel),
            refresh_per_second=int(self.config.get("refresh_per_second", 2)),
            transient=False,
        ) as live:
            for epoch in range(start_epoch, self.epochs):
                epoch_start = time.perf_counter()
                train_loss = self._train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    progress,
                    batch_task,
                    epoch,
                )
                val_loss, val_metrics, class_rows = self._validate(
                    model,
                    val_loader,
                    criterion,
                )
                progress.advance(epoch_task)
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
                        self.paths["checkpoint_path"],
                        model,
                        model_name=self.model_name,
                        config=self.config,
                    )
                    messages.append(f"best loss → {self.paths['checkpoint_path']}")
                foreground_dice = float(val_metrics["mean_foreground_dice"])
                if np.isfinite(foreground_dice) and foreground_dice > best_dice:
                    best_dice = foreground_dice
                    save_checkpoint(
                        self.paths["dice_checkpoint_path"],
                        model,
                        model_name=self.model_name,
                        config=self.config,
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
                checkpoint_panel = Panel(
                    "\n".join(messages),
                    title="Checkpoint",
                    border_style="green",
                )
                live.update(
                    Group(
                        progress,
                        _metrics_table(
                            epoch=epoch + 1,
                            epochs=self.epochs,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            metrics=val_metrics,
                        ),
                        checkpoint_panel,
                    )
                )
        print_success(
            f"Segmentation training complete; research artifacts are in {self.paths['output_dir']}"
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
        progress: Progress,
        batch_task,
        epoch: int,
    ) -> float:
        model.train()
        running_loss = 0.0
        sample_count = 0
        progress.reset(batch_task)
        progress.update(batch_task, description=f"[cyan]Epoch {epoch + 1} batches")
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(images)
            sample_count += len(images)
            progress.advance(batch_task)
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
    TrainSegmentationModel(config).execute()


def smoke_test_segmentation(config: dict[str, Any]) -> None:
    model_name = resolve_model_name(config)
    batch = int(config["batch_size"])
    width, height = config["input_size"]
    device = config["device"]
    num_classes = int(config.get("num_classes", 2))
    print_info(
        f"Running segmentation test on device={device} | input={width}x{height} "
        f"| batch={batch} | classes={num_classes}"
    )
    model = build_segmentation_model(model_name, config, num_classes=num_classes).to(device)
    channels = 3 if config.get("colored", True) else 1
    output = model(torch.randn(batch, channels, height, width).to(device))
    print_success("Test completed successfully!")
    print_info(f"Output tensor shape: {list(output.shape)}")
    table = Table(title="Segmentation Model Output", show_header=True)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="magenta")
    for index, value in enumerate(output.flatten().tolist()[:16]):
        table.add_row(str(index), f"{value:.6f}")
    console.print(table)


def _build_progress(*, epochs: int, train_loader_size: int) -> Progress:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    progress.add_task("[magenta]Epoch Progress", total=epochs)
    progress.add_task("[cyan]Batch Progress", total=train_loader_size)
    return progress


def _metrics_table(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, float],
) -> Table:
    table = Table(title=f"Epoch {epoch}/{epochs}", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    for label, value in (
        ("Train Loss", train_loss),
        ("Validation Loss", val_loss),
        ("Pixel Accuracy", metrics["pixel_accuracy"]),
        ("Macro Dice", metrics["macro_dice"]),
        ("Foreground Dice", metrics["mean_foreground_dice"]),
        ("Mean IoU", metrics["mean_iou"]),
        ("Foreground IoU", metrics["mean_foreground_iou"]),
    ):
        table.add_row(label, f"{value:.6f}")
    return table
