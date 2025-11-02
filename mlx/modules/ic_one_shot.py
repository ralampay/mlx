import os
import random
import math

import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import typer
from torchvision import transforms, datasets
import torchvision.transforms.v2 as T
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
from mlx.modules.datasets.one_shot_pair_dataset import OneShotPairDataset

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
        "colored": True,
        "num_pairs": 100
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
    elif config["action"] == "infer-image":
        _infer_image(net, config)
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
    device          = config["device"]
    dataset_path    = config["dataset_path"]
    batch_size      = config.get("batch_size", 4)
    epochs          = config.get("epochs", 20)
    lr              = config.get("lr", 1e-4)
    input_size      = config.get("input_size", (105, 105))
    colored         = config.get("colored", True)
    refresh_rate    = config.get("refresh_per_second", 2)

    checkpoint_dir = os.path.join(dataset_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    typer.secho(
        f"Starting training on device={device} for {epochs} epochs",
        fg=typer.colors.BRIGHT_YELLOW, bold=True
    )

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
    colored = config["colored"]

    typer.secho(
        f"Running test on device={device} | input={h}x{w} | batch={batch}",
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Assume colored
    input_channel_size = 3 if colored else 1
    x1 = torch.randn(batch, input_channel_size, h, w).to(device)
    x2 = torch.randn(batch, input_channel_size, h, w).to(device)

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
    device          = config.get("device", "cpu")
    test_path       = config["dataset_path"] # use dataset_path pointing to test folder
    model_path      = config["model_path"]
    batch_size      = config.get("batch_size", 2)
    img_size        = config.get("img_size", (105, 105))
    colored         = config.get("colored", True)
    embedding_size  = config.get("embedding_size", 4096)
    num_pairs       = config.get("num_pairs", 2000)

    # --- Instantiate and load model ---
    console.print(f"[cyan]Loading model from[/cyan] [bold]{model_path}[/bold] ...")
    net = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    #dataset = datasets.ImageFolder(test_path, transform=transform)
    dataset = OneShotPairDataset(
        test_path,
        input_size=img_size,
        colored=colored,
        n_pairs_per_class=num_pairs
    )

    console.print(f"[green]Loaded {len(dataset)} test images from {test_path}[/green]")

    # Dataset already provides pairs
    pairs_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Evaluate pairs ---
    preds, probs, targets = [], [], []
    with torch.no_grad():
        for img1, img2, target in tqdm(pairs_loader, desc="Evaluating pairs"):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            out = net(img1, img2)
            prob = torch.sigmoid(out).item()
            preds.append(1 if prob > 0.5 else 0)
            targets.append(target.item())


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

def _infer_image(model, config):
    """
    Perform inference for a given input image against a reference dataset using OpenCV.
    Consistent preprocessing with OneShotPairDataset.
    """
    model.eval()

    device          = config.get("device", "cpu")
    img_size        = config.get("img_size", (105, 105))
    colored         = config.get("colored", True)
    input_img_path  = config["input_img"]
    dataset_path    = config["dataset_path"]

    # --- helper to preprocess like OneShotPairDataset
    def preprocess(img_path):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        if not colored:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, img_size)
            img = np.expand_dims(img, axis=0)  # (1, H, W)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, C, H, W)
        return tensor

    def get_embedding(img_path):
        with torch.no_grad():
            tensor = preprocess(img_path)
            emb = model.embedding(tensor)
        return emb

    # --- get embedding of input image
    input_emb = get_embedding(input_img_path)

    # --- iterate over dataset images
    best_match = None
    min_distance = float("inf")
    all_scores = []

    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".PNG")):
                ref_path = os.path.join(root, fname)
                try:
                    ref_emb = get_embedding(ref_path)
                except Exception as e:
                    print(f"Skipping {ref_path}: {e}")
                    continue

                dist = F.pairwise_distance(input_emb, ref_emb).item()

                # --- determine label name
                if os.path.basename(root) != "" and root != dataset_path:
                    label = os.path.basename(root)
                else:
                    # fallback: try to infer from filename (before extension)
                    label = os.path.splitext(fname)[0]

                all_scores.append((label, ref_path, dist))

                if dist < min_distance:
                    min_distance = dist
                    best_match = (label, ref_path)

    # --- sort matches
    all_scores.sort(key=lambda x: x[2])

    # --- format result
    result = {
        "input_image": input_img_path,
        "best_match_label": best_match[0] if best_match else None,
        "best_match_path": best_match[1] if best_match else None,
        "distance": min_distance,
        "top_matches": all_scores[:10],
    }

    # --- Display results
    _display_inference_results(result)

    return result

def _display_inference_results(result):
    """
    Display inference results as:
    1. A composite grid of all matches (sorted by distance)
    2. A side-by-side window showing input vs best match
    """

    input_img = result["input_image"]
    all_matches = result["top_matches"]
    best_label = result["best_match_label"]
    best_path = result["best_match_path"]
    best_distance = result["distance"]

    # --- Rich Table
    table = Table(title="🔍 Inference Results (All Samples)", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", justify="center", style="magenta")
    table.add_column("Image Path", justify="left")
    table.add_column("Distance", justify="center", style="green")

    for i, (label, path, dist) in enumerate(all_matches, start=1):
        table.add_row(str(i), label, path, f"{dist:.4f}")

    console.print(table)
    typer.echo(f"\n✅ Best match: {best_label} (distance={best_distance:.4f})")

    # --- Helper for header bar
    def draw_header_bar(img, text):
        """
        Draw a black header bar with white text on top of the image.
        Ensures text is always visible and fully rendered.
        """
        if img is None:
            return img

        # --- Ensure image is in 3-channel BGR
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Dynamic scaling (adjusted empirically for small Omniglot-like images)
        font_scale = max(0.5, min(1.0, img.shape[1] / 250.0))
        thickness = max(1, int(img.shape[1] / 400))

        # --- Compute text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # --- Define bar height with padding above/below
        top_padding = 8
        bottom_padding = 6
        bar_height = text_h + baseline + top_padding + bottom_padding

        # --- Create black bar
        bar = np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8)

        # --- Text position (x: 10 px from left, y: baseline + padding + text height)
        y = top_padding + text_h

        # --- Draw text clearly on the bar
        cv2.putText(
            bar,
            text,
            (10, y),
            font,
            font_scale,
            (255, 255, 255),  # white text
            thickness,
            cv2.LINE_AA,
        )

        # --- Combine vertically
        combined = np.vstack((bar, img))
        return combined

    # --- Load all images
    imgs = []
    input_display = cv2.imread(input_img)
    if input_display is not None:
        input_display = draw_header_bar(input_display, "INPUT")
        imgs.append(input_display)

    for label, ref_path, dist in all_matches:
        ref_display = cv2.imread(ref_path)
        if ref_display is None:
            continue
        caption = f"{label} - dist {dist:.4f}"
        ref_display = draw_header_bar(ref_display, caption)
        imgs.append(ref_display)

    if not imgs:
        typer.echo("⚠️ No images to display.")
        return

    # --- Resize to consistent height
    target_height = 200
    resized_imgs = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = target_height / h
        resized = cv2.resize(img, (int(w * scale), target_height + 40))
        resized_imgs.append(resized)

    # --- Make grid of all results
    num_cols = 5
    num_rows = math.ceil(len(resized_imgs) / num_cols)
    row_imgs = []
    for i in range(num_rows):
        row = resized_imgs[i * num_cols:(i + 1) * num_cols]
        while len(row) < num_cols:
            row.append(np.zeros_like(resized_imgs[0]))
        row_imgs.append(np.hstack(row))
    grid = np.vstack(row_imgs)

    cv2.imshow("Inference Comparison (All Samples)", grid)

    # --- Separate window: input vs best match
    input_full = cv2.imread(input_img)
    best_full = cv2.imread(best_path)

    if input_full is not None and best_full is not None:
        input_full = draw_header_bar(input_full, "INPUT")
        best_full = draw_header_bar(best_full, f"{best_label} - dist {best_distance:.4f}")

        # Resize both to same height for side-by-side view
        h1, w1 = input_full.shape[:2]
        h2, w2 = best_full.shape[:2]
        target_height = min(400, max(h1, h2))
        input_resized = cv2.resize(input_full, (int(w1 * target_height / h1), target_height))
        best_resized = cv2.resize(best_full, (int(w2 * target_height / h2), target_height))

        side_by_side = np.hstack((input_resized, best_resized))
        cv2.imshow("Best Match Comparison", side_by_side)

    typer.echo("\nPress any key on an image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
