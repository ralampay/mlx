from __future__ import annotations

from pathlib import Path
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
from mlx.modes.image_classification.data import (
    load_one_shot_datasets,
    load_standard_classification_datasets,
)
from mlx.modes.image_classification.models import (
    build_image_classification_model,
    model_family_for,
)
from mlx.modes.image_classification.utils import (
    resolve_model_name,
    resolve_train_output_path,
    save_checkpoint,
)


def train_image_classification(config: dict[str, Any]) -> None:
    model_name = resolve_model_name(config)
    family = model_family_for(model_name)
    if family == "one-shot":
        _train_one_shot(model_name, config)
        return
    _train_standard(model_name, config)


def smoke_test_image_classification(config: dict[str, Any]) -> None:
    model_name = resolve_model_name(config)
    family = model_family_for(model_name)
    if family == "one-shot":
        _test_one_shot(model_name, config)
        return
    _test_standard(model_name, config)


def _train_one_shot(model_name: str, config: dict[str, Any]) -> None:
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 20)
    learning_rate = config.get("lr") or 1e-4
    input_size = config.get("input_size", (105, 105))
    colored = config.get("colored", True)
    refresh_rate = config.get("refresh_per_second", 2)
    output_path = resolve_train_output_path(config)

    print_info(f"Starting one-shot training on device={device} for {epochs} epochs")

    model = build_image_classification_model(model_name, config).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset, val_dataset = load_one_shot_datasets(
        dataset_path,
        input_size=input_size,
        colored=colored,
        n_pairs_per_class=config.get("num_pairs", 100),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    prev_train_loss = None
    prev_val_loss = None
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    progress = _build_progress(epochs=epochs, train_loader_size=len(train_loader))
    epoch_task, batch_task = _progress_tasks(progress, epochs=epochs)

    with Live(Group(progress, last_saved_panel), refresh_per_second=refresh_rate, transient=False) as live:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            progress.reset(batch_task)
            progress.update(epoch_task, description=f"[magenta]Epoch {epoch + 1}/{epochs}")

            for batch_index, (img1, img2, label) in enumerate(train_loader, start=1):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(img1, img2)
                loss = criterion(output, label.unsqueeze(1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress.advance(batch_task)
                progress.update(batch_task, description=f"[cyan]Batch {batch_index}/{len(train_loader)}")

            avg_train_loss = running_loss / len(train_loader)
            progress.advance(epoch_task)
            avg_val_loss = _validate_one_shot(model, val_loader, criterion, device)

            metrics_table = _build_loss_table(
                epoch=epoch,
                epochs=epochs,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                prev_train_loss=prev_train_loss,
                prev_val_loss=prev_val_loss,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    output_path,
                    model,
                    model_name=model_name,
                    family="one-shot",
                    config=config,
                )
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

    print_success("One-shot training complete!")


def _train_standard(model_name: str, config: dict[str, Any]) -> None:
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 20)
    learning_rate = config.get("lr") or 1e-3
    input_size = config.get("input_size", (224, 224))
    colored = config.get("colored", True)
    refresh_rate = config.get("refresh_per_second", 2)
    output_path = resolve_train_output_path(config)

    print_info(f"Starting standard classification training on device={device} for {epochs} epochs")
    train_dataset, val_dataset, classes = load_standard_classification_datasets(
        dataset_path,
        input_size=input_size,
        colored=colored,
    )
    model = build_image_classification_model(
        model_name,
        config,
        num_classes=len(classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    prev_train_loss = None
    prev_val_loss = None
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    progress = _build_progress(epochs=epochs, train_loader_size=len(train_loader))
    epoch_task, batch_task = _progress_tasks(progress, epochs=epochs)

    with Live(Group(progress, last_saved_panel), refresh_per_second=refresh_rate, transient=False) as live:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            progress.reset(batch_task)
            progress.update(epoch_task, description=f"[magenta]Epoch {epoch + 1}/{epochs}")

            for batch_index, (images, targets) in enumerate(train_loader, start=1):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress.advance(batch_task)
                progress.update(batch_task, description=f"[cyan]Batch {batch_index}/{len(train_loader)}")

            avg_train_loss = running_loss / len(train_loader)
            progress.advance(epoch_task)
            avg_val_loss, val_accuracy = _validate_standard(model, val_loader, criterion, device)
            metrics_table = _build_standard_metrics_table(
                epoch=epoch,
                epochs=epochs,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                prev_train_loss=prev_train_loss,
                prev_val_loss=prev_val_loss,
                val_accuracy=val_accuracy,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    output_path,
                    model,
                    model_name=model_name,
                    family="standard",
                    config=config,
                    classes=classes,
                )
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

    print_success("Standard classification training complete!")


def _test_one_shot(model_name: str, config: dict[str, Any]) -> None:
    batch = config["batch_size"]
    height, width = config["input_size"]
    device = config["device"]
    colored = config["colored"]

    print_info(f"Running one-shot test on device={device} | input={height}x{width} | batch={batch}")
    model = build_image_classification_model(model_name, config).to(device)

    channels = 3 if colored else 1
    x1 = torch.randn(batch, channels, height, width).to(device)
    x2 = torch.randn(batch, channels, height, width).to(device)
    output = model(x1, x2)

    _render_test_output("One-Shot Model Output", output)


def _test_standard(model_name: str, config: dict[str, Any]) -> None:
    batch = config["batch_size"]
    height, width = config["input_size"]
    device = config["device"]
    colored = config["colored"]
    num_classes = config.get("num_classes", 4)

    print_info(
        f"Running standard classification test on device={device} | input={height}x{width} | batch={batch}"
    )
    model = build_image_classification_model(
        model_name,
        config,
        num_classes=num_classes,
    ).to(device)

    channels = 3 if colored else 1
    x = torch.randn(batch, channels, height, width).to(device)
    output = model(x)
    _render_test_output("Classification Model Output", output)


def _render_test_output(title: str, output: torch.Tensor) -> None:
    print_success("Test completed successfully!")
    print_info(f"Output tensor shape: {list(output.shape)}")

    table = Table(title=title, show_header=True)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="magenta")
    for index, value in enumerate(output.flatten().tolist()[:16]):
        table.add_row(str(index), f"{value:.6f}")
    console.print(table)


def _validate_one_shot(model, val_loader, criterion, device: str) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = criterion(output, label.unsqueeze(1))
            val_loss += loss.item()
    return val_loss / len(val_loader)


def _validate_standard(model, val_loader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())

    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


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


def _progress_tasks(progress: Progress, *, epochs: int) -> tuple[int, int]:
    return progress.task_ids[0], progress.task_ids[1]


def _build_loss_table(
    *,
    epoch: int,
    epochs: int,
    avg_train_loss: float,
    avg_val_loss: float,
    prev_train_loss: float | None,
    prev_val_loss: float | None,
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
    return table


def _build_standard_metrics_table(
    *,
    epoch: int,
    epochs: int,
    avg_train_loss: float,
    avg_val_loss: float,
    prev_train_loss: float | None,
    prev_val_loss: float | None,
    val_accuracy: float,
) -> Table:
    table = _build_loss_table(
        epoch=epoch,
        epochs=epochs,
        avg_train_loss=avg_train_loss,
        avg_val_loss=avg_val_loss,
        prev_train_loss=prev_train_loss,
        prev_val_loss=prev_val_loss,
    )
    table.add_row("Val Accuracy", "-", f"{val_accuracy:.4f}", "-")
    return table


def _loss_delta(previous: float | None, current: float) -> str:
    if previous is None:
        return "-"
    if current < previous:
        return f"↓ {previous - current:.4f}"
    return f"↑ {current - previous:.4f}"
