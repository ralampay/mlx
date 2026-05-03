from __future__ import annotations

from typing import Any

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

from mlx.core.ui import console, print_info, print_success
from mlx.modes.segmentation.data import load_segmentation_datasets
from mlx.modes.segmentation.models import build_segmentation_model
from mlx.modes.segmentation.utils import (
    compute_dice_score,
    compute_mean_iou,
    compute_pixel_accuracy,
    resolve_model_name,
    resolve_train_output_path,
    save_checkpoint,
)


def train_segmentation(config: dict[str, Any]) -> None:
    model_name = resolve_model_name(config)
    device = config["device"]
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 50)
    learning_rate = config.get("lr") or 1e-3
    input_size = config.get("input_size", (256, 256))
    num_classes = int(config.get("num_classes", 2))
    colored = bool(config.get("colored", True))
    output_path = resolve_train_output_path(config)
    refresh_rate = config.get("refresh_per_second", 2)

    print_info(f"Starting segmentation training on device={device} for {epochs} epochs")
    train_dataset, val_dataset = load_segmentation_datasets(
        config["dataset_path"],
        input_size=input_size,
        num_classes=num_classes,
        colored=colored,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_segmentation_model(model_name, config, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    prev_train_loss = None
    prev_val_loss = None
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    progress = _build_progress(epochs=epochs, train_loader_size=len(train_loader))
    epoch_task, batch_task = progress.task_ids[0], progress.task_ids[1]

    with Live(Group(progress, last_saved_panel), refresh_per_second=refresh_rate, transient=False) as live:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            progress.reset(batch_task)
            progress.update(epoch_task, description=f"[magenta]Epoch {epoch + 1}/{epochs}")

            for batch_index, (images, masks) in enumerate(train_loader, start=1):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress.advance(batch_task)
                progress.update(batch_task, description=f"[cyan]Batch {batch_index}/{len(train_loader)}")

            avg_train_loss = running_loss / max(1, len(train_loader))
            progress.advance(epoch_task)
            avg_val_loss, val_accuracy, val_dice, val_iou = _validate_segmentation(
                model,
                val_loader,
                criterion,
                device,
                num_classes=num_classes,
            )
            metrics_table = _build_metrics_table(
                epoch=epoch,
                epochs=epochs,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                prev_train_loss=prev_train_loss,
                prev_val_loss=prev_val_loss,
                val_accuracy=val_accuracy,
                val_dice=val_dice,
                val_iou=val_iou,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(output_path, model, model_name=model_name, config=config)
                last_saved_panel = Panel(
                    f"[green]Saved new best model at {output_path}[/]",
                    title="Checkpoint",
                    border_style="green",
                )
            else:
                last_saved_panel = Panel("No improvement", title="Checkpoint", border_style="dim")

            live.update(Group(progress, metrics_table, last_saved_panel))
            prev_train_loss = avg_train_loss
            prev_val_loss = avg_val_loss

    print_success("Segmentation training complete!")


def smoke_test_segmentation(config: dict[str, Any]) -> None:
    model_name = resolve_model_name(config)
    batch = config["batch_size"]
    width, height = config["input_size"]
    device = config["device"]
    colored = config["colored"]
    num_classes = int(config.get("num_classes", 2))

    print_info(
        f"Running segmentation test on device={device} | input={width}x{height} | batch={batch} | classes={num_classes}"
    )
    model = build_segmentation_model(model_name, config, num_classes=num_classes).to(device)
    channels = 3 if colored else 1
    x = torch.randn(batch, channels, height, width).to(device)
    output = model(x)

    print_success("Test completed successfully!")
    print_info(f"Output tensor shape: {list(output.shape)}")

    table = Table(title="Segmentation Model Output", show_header=True)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="magenta")
    for index, value in enumerate(output.flatten().tolist()[:16]):
        table.add_row(str(index), f"{value:.6f}")
    console.print(table)


def _validate_segmentation(
    model,
    val_loader,
    criterion,
    device: str,
    *,
    num_classes: int,
) -> tuple[float, float, float, float]:
    model.eval()
    val_loss = 0.0
    accuracies = []
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            predictions = logits.argmax(dim=1)
            accuracies.append(compute_pixel_accuracy(predictions, masks))
            dice_scores.append(compute_dice_score(predictions, masks, num_classes))
            iou_scores.append(compute_mean_iou(predictions, masks, num_classes))

    avg_loss = val_loss / max(1, len(val_loader))
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    return avg_loss, avg_accuracy, avg_dice, avg_iou


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


def _build_metrics_table(
    *,
    epoch: int,
    epochs: int,
    avg_train_loss: float,
    avg_val_loss: float,
    prev_train_loss: float | None,
    prev_val_loss: float | None,
    val_accuracy: float,
    val_dice: float,
    val_iou: float,
) -> Table:
    table = Table(title=f"Epoch {epoch + 1}/{epochs}", show_lines=True)
    table.add_column("Metric", justify="center", style="cyan")
    table.add_column("Previous", justify="center", style="yellow")
    table.add_column("Current", justify="center", style="magenta")
    table.add_column("Delta", justify="center", style="bright_black")
    table.add_row(
        "Train Loss",
        f"{prev_train_loss:.6f}" if prev_train_loss is not None else "-",
        f"{avg_train_loss:.6f}",
        _loss_delta(prev_train_loss, avg_train_loss),
    )
    table.add_row(
        "Val Loss",
        f"{prev_val_loss:.6f}" if prev_val_loss is not None else "-",
        f"{avg_val_loss:.6f}",
        _loss_delta(prev_val_loss, avg_val_loss),
    )
    table.add_row("Val Accuracy", "-", f"{val_accuracy:.4f}", "-")
    table.add_row("Val Dice", "-", f"{val_dice:.4f}", "-")
    table.add_row("Val mIoU", "-", f"{val_iou:.4f}", "-")
    return table


def _loss_delta(previous: float | None, current: float) -> str:
    if previous is None:
        return "-"
    if current < previous:
        return f"↓ {previous - current:.4f}"
    return f"↑ {current - previous:.4f}"
