import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import typer
from torchvision import transforms, datasets
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from rich.table import Table
from rich.console import Console, Group
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from mlx.modules.data_builder import load_ic_one_shot_dataset
from mlx.modules.ic.siamese_le_net import SiameseLeNet
from mlx.modules.data_builder import build_ic_one_shot
from mlx.utils import render_loss_plot

console = Console()

def run_ic_one_shot(model, **kwargs):
    defaults = {
        "action": "test",
        "device": "cpu",
        "embedding_size": 4096,
        "input_size": (105, 105),
        "batch_size": 1,
        "dataset_path": "",
        "epochs": 100,
        "refresh_per_second": 2,
        "colored": True
    }

     # Merge defaults with kwargs (user overrides)
    config = {**defaults, **kwargs}

    # Display configuration summary
    _print_config_summary(model, config)

    if model == "siamese-le-net":
        net = SiameseLeNet(
            colored=config["colored"], 
            embedding_size=config["embedding_size"]
        )
    else:
        raise ValueError(f"Invalid model {model}")

    if config["action"] == "test":
        _test_model(net, config)
    elif config["action"] == "train":
        _train_model(net, config)
    elif config["action"] == "benchmark":
        _benchmark_model(net, config)
    elif config["action"] == "build-dataset":
        build_ic_one_shot(config["dataset_path"])
    else:
        raise ValueError(f"Unsupported action {config['action']}")

def _print_config_summary(model: str, config: dict):
    """Print a nicely formatted configuration summary."""
    table = Table(title=f"Configuration for {model}", show_lines=True)
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)

def _train_model(net, config):
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 20)
    lr = config.get("lr", 1e-4)
    input_size = config.get("input_size", (105, 105))
    colored = config.get("colored", True)
    refresh_rate = config.get("refresh_per_second", 2)

    checkpoint_dir = os.path.join(dataset_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    typer.secho(f"Starting training on device={device} for {epochs} epochs",
                fg=typer.colors.BRIGHT_YELLOW, bold=True)

    net = net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Load datasets
    train_dataset, val_dataset = load_ic_one_shot_dataset(
        dataset_path, input_size=input_size, colored=colored
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    prev_train_loss, prev_val_loss = None, None
    last_saved_panel = Panel("No model saved yet", border_style="dim")

    # Rich progress setup
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

    # Initial table placeholder
    table = Table(title="Training Status", show_lines=True)
    table.add_column("Metric", justify="center", style="cyan")
    table.add_column("Previous", justify="center", style="yellow")
    table.add_column("Current", justify="center", style="magenta")
    table.add_column("Δ", justify="center", style="bright_black")
    table.add_row("Train Loss", "-", "-", "-")
    table.add_row("Val Loss", "-", "-", "-")

    layout = Group(progress, table, last_saved_panel)

    # ---- Unified Live render ----
    with Live(layout, refresh_per_second=refresh_rate, transient=False) as live:
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0

            progress.reset(batch_task)
            progress.update(epoch_task, description=f"[magenta]Epoch {epoch + 1}/{epochs}")

            for batch_idx, (img1, img2, label) in enumerate(train_loader, start=1):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                optimizer.zero_grad()
                output = net(img1, img2)
                loss = criterion(output, label.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.advance(batch_task)
                progress.update(batch_task, description=f"[cyan]Batch {batch_idx}/{len(train_loader)}")

            avg_train_loss = running_loss / len(train_loader)
            progress.advance(epoch_task)

            # ---- VALIDATE ----
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img1, img2, label in val_loader:
                    img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                    output = net(img1, img2)
                    loss = criterion(output, label.unsqueeze(1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # ---- Calculate deltas ----
            train_delta = (
                f"↓ {prev_train_loss - avg_train_loss:.4f}"
                if prev_train_loss and avg_train_loss < prev_train_loss else
                (f"↑ {avg_train_loss - prev_train_loss:.4f}" if prev_train_loss else "-")
            )
            val_delta = (
                f"↓ {prev_val_loss - avg_val_loss:.4f}"
                if prev_val_loss and avg_val_loss < prev_val_loss else
                (f"↑ {avg_val_loss - prev_val_loss:.4f}" if prev_val_loss else "-")
            )

            # ---- Update table ----
            new_table = Table(title=f"Epoch {epoch + 1}/{epochs}", show_lines=True)
            new_table.add_column("Metric", justify="center", style="cyan")
            new_table.add_column("Previous", justify="center", style="yellow")
            new_table.add_column("Current", justify="center", style="magenta")
            new_table.add_column("Δ", justify="center", style="bright_black")
            new_table.add_row("Train Loss", 
                              f"{prev_train_loss:.6f}" if prev_train_loss else "-", 
                              f"{avg_train_loss:.6f}", train_delta)
            new_table.add_row("Val Loss", 
                              f"{prev_val_loss:.6f}" if prev_val_loss else "-", 
                              f"{avg_val_loss:.6f}", val_delta)

            # ---- Save best model ----
            saved_msg = "No improvement"
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(checkpoint_dir, f"best_epoch_{epoch+1}.pt")
                torch.save(net.state_dict(), model_path)
                saved_msg = f"[green]✅ Saved new best model at {model_path}[/]"

            last_saved_panel = Panel(saved_msg, title="Checkpoint", border_style="green" if "✅" in saved_msg else "dim")

            # ---- Update live layout ----
            live.update(Group(progress, new_table, last_saved_panel))

            prev_train_loss, prev_val_loss = avg_train_loss, avg_val_loss

    typer.secho("\nTraining complete!", fg=typer.colors.BRIGHT_GREEN, bold=True)


def _test_model(net, config):
    """Run test with random tensors using resolved config."""
    batch   = config["batch_size"]
    h, w    = config["input_size"]
    device  = config["device"]

    typer.secho(
        f"Running test on device={device} | input={h}x{w} | batch={batch}",
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Assume colored
    x1 = torch.randn(batch, 3, h, w).to(device)
    x2 = torch.randn(batch, 3, h, w).to(device)

    out = net(x1, x2)

    typer.secho("\nTest completed successfully!", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho(f"Output tensor shape: {list(out.shape)}\n", fg=typer.colors.BRIGHT_GREEN)

    # Display output in compact table
    table = Table(title="Model Output", show_header=True)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="magenta")

    for i, val in enumerate(out.flatten().tolist()):
        table.add_row(str(i), f"{val:.6f}")

    console.print(table)

def _benchmark_model(model, config):
    """
    Benchmarks a trained one-shot model using embeddings and cosine similarity.

    Args:
        model: The model class to instantiate (e.g., SiameseLeNet)
        config (dict): Must include:
            - "model_path": path to the saved .pt file
            - "test_path": path to test dataset
            - "device": "cuda" or "cpu"
            - Optional: "batch_size", "embedding_size", "img_size"
    """

    console = Console()
    console.rule("[bold blue]Benchmarking Model[/bold blue]")

    # --- Load config values ---
    device = config.get("device", "cpu")
    test_path = config["dataset_path"] # use dataset_path pointing to test folder
    model_path = config["model_path"]
    batch_size = config.get("batch_size", 2)
    img_size = config.get("img_size", (105, 105))
    colored = config.get("colored", True)
    embedding_size = config.get("embedding_size", 4096)
    num_pairs = config.get("num_pairs", 2000)

    # --- Instantiate and load model ---
    console.print(f"[cyan]Loading model from[/cyan] [bold]{model_path}[/bold] ...")
    net = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    # --- Prepare dataset ---
    compose_pipeline = []

    if not colored:
        compose_pipeline.append(
            transforms.Grayscale()
        )

    compose_pipeline.append(transforms.Resize(img_size))
    compose_pipeline.append(transforms.ToTensor())
    

    transform = transforms.Compose(compose_pipeline)

    dataset = datasets.ImageFolder(test_path, transform=transform)

    console.print(f"[green]Loaded {len(dataset)} test images from {test_path}[/green]")

    # --- Build label index for pairing ---
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices.setdefault(label, []).append(idx)

    # --- Generate positive and negative pairs ---
    pairs, targets = [], []
    labels = list(label_to_indices.keys())
    for _ in range(num_pairs):
        # positive
        c = random.choice(labels)
        i1, i2 = random.sample(label_to_indices[c], 2)
        pairs.append((i1, i2))
        targets.append(1)
        # negative
        c1, c2 = random.sample(labels, 2)
        i1 = random.choice(label_to_indices[c1])
        i2 = random.choice(label_to_indices[c2])
        pairs.append((i1, i2))
        targets.append(0)

    # --- Evaluate pairs ---
    preds, probs = [], []
    with torch.no_grad():
        for (i1, i2), target in tqdm(zip(pairs, targets), total=len(pairs), desc="Evaluating pairs"):
            img1, _ = dataset[i1]
            img2, _ = dataset[i2]
            img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
            out = net(img1, img2)
            prob = torch.sigmoid(out).item()
            preds.append(1 if prob > 0.5 else 0)
            probs.append(prob)

    # --- Metrics ---
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    # --- Display ---
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Score", justify="right")
    table.add_row("Accuracy", f"{acc:.4f}")
    table.add_row("Precision", f"{prec:.4f}")
    table.add_row("Recall", f"{rec:.4f}")
    table.add_row("F1-score", f"{f1:.4f}")

    console.print(table)
    console.rule("[green]Benchmarking Complete[/green]")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
