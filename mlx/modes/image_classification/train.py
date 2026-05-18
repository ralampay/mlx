from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import torch
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
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
    resolve_train_output_paths,
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
    use_best = bool(config.get("use_best", False))
    verbose = bool(config.get("verbose", False))
    output_paths = resolve_train_output_paths(config, model_name=model_name)
    checkpoint_path = output_paths["checkpoint_path"]
    training_csv_path = output_paths["training_csv_path"]
    _initialize_training_csv(training_csv_path)

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
    epoch_log = _build_epoch_log()
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    progress = _build_progress(epochs=epochs, train_loader_size=len(train_loader)) if verbose else None
    epoch_task, batch_task = _progress_tasks(progress) if progress is not None else (None, None)

    live = (
        Live(_render_training_view(epoch_log, progress, last_saved_panel), refresh_per_second=refresh_rate, transient=False)
        if verbose and progress is not None
        else None
    )

    with (live or _nullcontext()):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            if progress is not None and batch_task is not None and epoch_task is not None:
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
                if progress is not None and batch_task is not None:
                    progress.advance(batch_task)
                    progress.update(batch_task, description=f"[cyan]Batch {batch_index}/{len(train_loader)}")

            avg_train_loss = running_loss / len(train_loader)
            if progress is not None and epoch_task is not None:
                progress.advance(epoch_task)
            avg_val_loss, val_metrics = _validate_one_shot(model, val_loader, criterion, device)
            _append_training_csv_row(
                training_csv_path,
                epoch=epoch + 1,
                loss=avg_train_loss,
                metric=avg_val_loss,
            )
            epoch_log = _append_epoch_log(
                epoch_log,
                epoch=epoch + 1,
                epochs=epochs,
                values=[
                    ("loss", avg_train_loss),
                    ("val_loss", avg_val_loss),
                    ("accuracy", val_metrics["accuracy"]),
                    ("precision", val_metrics["precision"]),
                    ("recall", val_metrics["recall"]),
                    ("f1", val_metrics["f1"]),
                ],
            )

            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
            saved_checkpoint = not use_best or improved
            if saved_checkpoint:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    model_name=model_name,
                    family="one-shot",
                    config=config,
                )
                checkpoint_message = (
                    f"Saved new best model at {checkpoint_path}"
                    if use_best
                    else f"Saved epoch {epoch + 1} model at {checkpoint_path}"
                )
                last_saved_panel = Panel(
                    f"[green]{checkpoint_message}[/]",
                    title="Checkpoint",
                    border_style="green",
                )
            else:
                checkpoint_message = None
                last_saved_panel = Panel("No improvement", title="Checkpoint", border_style="dim")

            if live is not None and progress is not None:
                live.update(_render_training_view(epoch_log, progress, last_saved_panel))
            else:
                _print_epoch_summary(
                    epoch=epoch + 1,
                    epochs=epochs,
                    values=[
                        ("loss", avg_train_loss),
                        ("val_loss", avg_val_loss),
                        ("accuracy", val_metrics["accuracy"]),
                        ("precision", val_metrics["precision"]),
                        ("recall", val_metrics["recall"]),
                        ("f1", val_metrics["f1"]),
                    ],
                    checkpoint_path=checkpoint_path,
                    saved_checkpoint=saved_checkpoint,
                    checkpoint_message=checkpoint_message,
                )

    print_success("One-shot training complete!")


def _train_standard(model_name: str, config: dict[str, Any]) -> None:
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 20)
    learning_rate = config.get("lr") or 1e-3
    input_size = config.get("input_size", (224, 224))
    colored = config.get("colored", True)
    apply_transformations = bool(config.get("apply_transformations", False))
    refresh_rate = config.get("refresh_per_second", 2)
    use_best = bool(config.get("use_best", False))
    verbose = bool(config.get("verbose", False))
    output_paths = resolve_train_output_paths(config, model_name=model_name)
    checkpoint_path = output_paths["checkpoint_path"]
    training_csv_path = output_paths["training_csv_path"]
    _initialize_training_csv(training_csv_path)

    print_info(f"Starting standard classification training on device={device} for {epochs} epochs")
    train_dataset, val_dataset, classes = load_standard_classification_datasets(
        dataset_path,
        input_size=input_size,
        colored=colored,
        apply_transformations=apply_transformations,
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
    epoch_log = _build_epoch_log()
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    progress = _build_progress(epochs=epochs, train_loader_size=len(train_loader)) if verbose else None
    epoch_task, batch_task = _progress_tasks(progress) if progress is not None else (None, None)

    live = (
        Live(_render_training_view(epoch_log, progress, last_saved_panel), refresh_per_second=refresh_rate, transient=False)
        if verbose and progress is not None
        else None
    )

    with (live or _nullcontext()):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            if progress is not None and batch_task is not None and epoch_task is not None:
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
                if progress is not None and batch_task is not None:
                    progress.advance(batch_task)
                    progress.update(batch_task, description=f"[cyan]Batch {batch_index}/{len(train_loader)}")

            avg_train_loss = running_loss / len(train_loader)
            if progress is not None and epoch_task is not None:
                progress.advance(epoch_task)
            avg_val_loss, val_metrics = _validate_standard(model, val_loader, criterion, device)
            _append_training_csv_row(
                training_csv_path,
                epoch=epoch + 1,
                loss=avg_train_loss,
                metric=val_metrics["accuracy"],
            )
            epoch_log = _append_epoch_log(
                epoch_log,
                epoch=epoch + 1,
                epochs=epochs,
                values=[
                    ("loss", avg_train_loss),
                    ("val_loss", avg_val_loss),
                    ("accuracy", val_metrics["accuracy"]),
                    ("precision", val_metrics["precision"]),
                    ("recall", val_metrics["recall"]),
                    ("f1", val_metrics["f1"]),
                ],
            )

            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
            saved_checkpoint = not use_best or improved
            if saved_checkpoint:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    model_name=model_name,
                    family="standard",
                    config=config,
                    classes=classes,
                )
                checkpoint_message = (
                    f"Saved new best model at {checkpoint_path}"
                    if use_best
                    else f"Saved epoch {epoch + 1} model at {checkpoint_path}"
                )
                last_saved_panel = Panel(
                    f"[green]{checkpoint_message}[/]",
                    title="Checkpoint",
                    border_style="green",
                )
            else:
                checkpoint_message = None
                last_saved_panel = Panel("No improvement", title="Checkpoint", border_style="dim")

            if live is not None and progress is not None:
                live.update(_render_training_view(epoch_log, progress, last_saved_panel))
            else:
                _print_epoch_summary(
                    epoch=epoch + 1,
                    epochs=epochs,
                    values=[
                        ("loss", avg_train_loss),
                        ("val_loss", avg_val_loss),
                        ("accuracy", val_metrics["accuracy"]),
                        ("precision", val_metrics["precision"]),
                        ("recall", val_metrics["recall"]),
                        ("f1", val_metrics["f1"]),
                    ],
                    checkpoint_path=checkpoint_path,
                    saved_checkpoint=saved_checkpoint,
                    checkpoint_message=checkpoint_message,
                )

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


def _validate_one_shot(model, val_loader, criterion, device: str) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    preds: list[int] = []
    targets_all: list[int] = []
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = criterion(output, label.unsqueeze(1))
            val_loss += loss.item()
            predictions = (output >= 0.5).float().squeeze(1)
            preds.extend(int(item) for item in predictions.cpu().tolist())
            targets_all.extend(int(item) for item in label.cpu().tolist())
    avg_loss = val_loss / len(val_loader)
    return avg_loss, _compute_classification_metrics(targets_all, preds)


def _validate_standard(model, val_loader, criterion, device: str) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    preds: list[int] = []
    targets_all: list[int] = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()
            predictions = logits.argmax(dim=1)
            preds.extend(int(item) for item in predictions.cpu().tolist())
            targets_all.extend(int(item) for item in targets.cpu().tolist())

    avg_loss = val_loss / len(val_loader)
    return avg_loss, _compute_classification_metrics(targets_all, preds)


def _compute_classification_metrics(targets: list[int], preds: list[int]) -> dict[str, float]:
    if not targets:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    labels = sorted(set(targets) | set(preds))
    total = len(targets)
    accuracy = sum(int(target == pred) for target, pred in zip(targets, preds, strict=True)) / total

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for label in labels:
        tp = sum(1 for target, pred in zip(targets, preds, strict=True) if target == label and pred == label)
        fp = sum(1 for target, pred in zip(targets, preds, strict=True) if target != label and pred == label)
        fn = sum(1 for target, pred in zip(targets, preds, strict=True) if target == label and pred != label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    num_labels = len(labels)
    return {
        "accuracy": accuracy,
        "precision": macro_precision / num_labels,
        "recall": macro_recall / num_labels,
        "f1": macro_f1 / num_labels,
    }


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


def _progress_tasks(progress: Progress) -> tuple[int, int]:
    return progress.task_ids[0], progress.task_ids[1]


def _initialize_training_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "loss", "metric"])

def _append_training_csv_row(csv_path: Path, *, epoch: int, loss: float, metric: float) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, f"{loss:.6f}", f"{metric:.6f}"])


def _build_epoch_log(*, max_entries: int = 12) -> deque[str]:
    return deque(maxlen=max_entries)


def _append_epoch_log(
    epoch_log: deque[str],
    *,
    epoch: int,
    epochs: int,
    values: list[tuple[str, float]],
) -> deque[str]:
    formatted_values = "  ".join(f"{label}: {value:.6f}" for label, value in values)
    epoch_log.append(f"Epoch {epoch}/{epochs}  {formatted_values}")
    return epoch_log


def _print_epoch_summary(
    *,
    epoch: int,
    epochs: int,
    values: list[tuple[str, float]],
    checkpoint_path: Path,
    saved_checkpoint: bool,
    checkpoint_message: str | None = None,
) -> None:
    formatted_values = "  ".join(f"{label}: {value:.6f}" for label, value in values)
    print_info(f"Epoch {epoch}/{epochs}  {formatted_values}")
    if saved_checkpoint:
        print_success(checkpoint_message or f"Saved model at {checkpoint_path}")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _render_training_view(epoch_log: deque[str], progress: Progress, last_saved_panel: Panel) -> Group:
    return Group(
        _render_epoch_log_panel(epoch_log),
        progress,
        last_saved_panel,
    )


def _render_epoch_log_panel(epoch_log: deque[str]) -> Panel:
    if epoch_log:
        body = Text("\n".join(epoch_log))
    else:
        body = Text("Waiting for completed epochs...", style="dim")
    return Panel(body, title="Epoch Results", border_style="blue")
