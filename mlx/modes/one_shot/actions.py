from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_success, print_warning
from mlx.modes.one_shot.data import (
    OneShotPairDataset,
    build_ic_one_shot,
    iter_dataset_images,
    load_ic_one_shot_dataset,
    load_image_tensor,
)
from mlx.modes.one_shot.models import SiameseLeNet
from mlx.modes.one_shot.presentation import display_inference_results


def build_model(model_name: str, config: dict[str, Any]):
    if model_name != "siamese-le-net":
        raise MLXUserError(f"Invalid model '{model_name}'.")
    return SiameseLeNet(
        colored=config["colored"],
        embedding_size=config["embedding_size"],
    )


def train_model(net, config: dict[str, Any]) -> None:
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 20)
    learning_rate = config.get("lr", 1e-4)
    input_size = config.get("input_size", (105, 105))
    colored = config.get("colored", True)
    refresh_rate = config.get("refresh_per_second", 2)

    checkpoint_dir = Path(dataset_path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"Starting training on device={device} for {epochs} epochs")

    net = net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_dataset, val_dataset = load_ic_one_shot_dataset(
        dataset_path,
        input_size=input_size,
        colored=colored,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    prev_train_loss = None
    prev_val_loss = None
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    epoch_task = progress.add_task("[magenta]Epoch Progress", total=epochs)
    batch_task = progress.add_task("[cyan]Batch Progress", total=len(train_loader))

    with Live(Group(progress, last_saved_panel), refresh_per_second=refresh_rate, transient=False) as live:
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            progress.reset(batch_task)
            progress.update(epoch_task, description=f"[magenta]Epoch {epoch + 1}/{epochs}")

            for batch_index, (img1, img2, label) in enumerate(train_loader, start=1):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                optimizer.zero_grad()
                output = net(img1, img2)
                loss = criterion(output, label.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.advance(batch_task)
                progress.update(batch_task, description=f"[cyan]Batch {batch_index}/{len(train_loader)}")

            avg_train_loss = running_loss / len(train_loader)
            progress.advance(epoch_task)

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img1, img2, label in val_loader:
                    img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                    output = net(img1, img2)
                    loss = criterion(output, label.unsqueeze(1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            metrics_table = _build_epoch_metrics_table(
                epoch=epoch,
                epochs=epochs,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                prev_train_loss=prev_train_loss,
                prev_val_loss=prev_val_loss,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = checkpoint_dir / f"best_epoch_{epoch + 1}.pt"
                torch.save(net.state_dict(), model_path)
                last_saved_panel = Panel(
                    f"[green]Saved new best model at {model_path}[/]",
                    title="Checkpoint",
                    border_style="green",
                )
            else:
                last_saved_panel = Panel("No improvement", title="Checkpoint", border_style="dim")

            live.update(Group(progress, metrics_table, last_saved_panel))
            prev_train_loss = avg_train_loss
            prev_val_loss = avg_val_loss

    print_success("Training complete!")


def test_model(net, config: dict[str, Any]) -> None:
    batch = config["batch_size"]
    height, width = config["input_size"]
    device = config["device"]
    colored = config["colored"]

    print_info(f"Running test on device={device} | input={height}x{width} | batch={batch}")

    channels = 3 if colored else 1
    x1 = torch.randn(batch, channels, height, width).to(device)
    x2 = torch.randn(batch, channels, height, width).to(device)
    output = net(x1, x2)

    print_success("Test completed successfully!")
    print_info(f"Output tensor shape: {list(output.shape)}")

    table = Table(title="Model Output", show_header=True)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="magenta")
    for index, value in enumerate(output.flatten().tolist()):
        table.add_row(str(index), f"{value:.6f}")
    console.print(table)


def benchmark_model(net, config: dict[str, Any]) -> dict[str, float]:
    console.rule("[bold blue]Benchmarking Model[/bold blue]")

    device = config.get("device", "cpu")
    test_path = config["dataset_path"]
    model_path = config["model_path"]
    img_size = config.get("img_size", (105, 105))
    colored = config.get("colored", True)
    num_pairs = config.get("num_pairs", 2000)

    if not model_path:
        raise MLXUserError("Benchmarking requires --model-path pointing to a checkpoint.")

    console.print(f"[cyan]Loading model from[/cyan] [bold]{model_path}[/bold] ...")
    net = net.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    net.eval()

    dataset = OneShotPairDataset(
        test_path,
        input_size=img_size,
        colored=colored,
        n_pairs_per_class=num_pairs,
    )
    console.print(f"[green]Loaded {len(dataset)} test images from {test_path}[/green]")

    pairs_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds = []
    targets = []
    with torch.no_grad():
        for img1, img2, target in tqdm(pairs_loader, desc="Evaluating pairs"):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            out = net(img1, img2)
            prob = torch.sigmoid(out).item()
            preds.append(1 if prob > 0.5 else 0)
            targets.append(target.item())

    results = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds),
        "recall": recall_score(targets, preds),
        "f1": f1_score(targets, preds),
    }

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Score", justify="right")
    table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    table.add_row("Precision", f"{results['precision']:.4f}")
    table.add_row("Recall", f"{results['recall']:.4f}")
    table.add_row("F1-score", f"{results['f1']:.4f}")
    console.print(table)
    console.rule("[green]Benchmarking Complete[/green]")
    return results


def infer_image(net, config: dict[str, Any]) -> dict[str, Any]:
    net.eval()

    device = config.get("device", "cpu")
    img_size = config.get("img_size", (105, 105))
    colored = config.get("colored", True)
    input_img_path = Path(config["input_img"])
    dataset_path = Path(config["dataset_path"])

    def embedding_for(image_path: Path) -> torch.Tensor:
        with torch.no_grad():
            tensor = load_image_tensor(image_path, input_size=img_size, colored=colored)
            tensor = tensor.unsqueeze(0).to(device)
            return net.embedding(tensor)

    input_embedding = embedding_for(input_img_path)
    best_match = None
    min_distance = float("inf")
    all_scores = []

    for reference_path in iter_dataset_images(dataset_path):
        try:
            reference_embedding = embedding_for(reference_path)
        except MLXUserError as exc:
            print_warning(f"Skipping {reference_path}: {exc}")
            continue

        distance = F.pairwise_distance(input_embedding, reference_embedding).item()
        label = reference_path.parent.name if reference_path.parent != dataset_path else reference_path.stem
        all_scores.append((label, reference_path, distance))

        if distance < min_distance:
            min_distance = distance
            best_match = (label, reference_path)

    all_scores.sort(key=lambda item: item[2])
    result = {
        "input_image": input_img_path,
        "best_match_label": best_match[0] if best_match else None,
        "best_match_path": best_match[1] if best_match else None,
        "distance": min_distance,
        "top_matches": all_scores[:10],
    }
    display_inference_results(result)
    return result


ACTION_HANDLERS = {
    "benchmark": benchmark_model,
    "build-dataset": lambda net, config: build_ic_one_shot(config["dataset_path"]),
    "infer-image": infer_image,
    "test": test_model,
    "train": train_model,
}


def _build_epoch_metrics_table(
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


def _loss_delta(previous: float | None, current: float) -> str:
    if previous is None:
        return "-"
    if current < previous:
        return f"↓ {previous - current:.4f}"
    return f"↑ {current - previous:.4f}"
