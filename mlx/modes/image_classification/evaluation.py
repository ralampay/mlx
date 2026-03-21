from __future__ import annotations

from typing import Any

import torch
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console
from mlx.modes.image_classification.data import (
    OneShotPairDataset,
    load_standard_classification_directory,
    resolve_evaluation_dir,
)
from mlx.modes.image_classification.utils import load_checkpoint_bundle


def benchmark_image_classification(config: dict[str, Any]) -> dict[str, float]:
    model, metadata = load_checkpoint_bundle(config)
    family = metadata["family"]
    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    if family == "one-shot":
        return _benchmark_one_shot(model, metadata, config, device)
    return _benchmark_standard(model, metadata, config, device)


def _benchmark_one_shot(model, metadata: dict[str, Any], config: dict[str, Any], device: str) -> dict[str, float]:
    test_path = resolve_evaluation_dir(config["dataset_path"])
    num_pairs = config.get("num_pairs", 2000)
    dataset = OneShotPairDataset(
        test_path,
        input_size=metadata["input_size"],
        colored=metadata["colored"],
        n_pairs_per_class=num_pairs,
    )
    console.print(f"[green]Loaded {len(dataset)} one-shot pairs from {test_path}[/green]")

    pairs_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds = []
    targets = []
    with torch.no_grad():
        for img1, img2, target in tqdm(pairs_loader, desc="Evaluating pairs"):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            out = model(img1, img2)
            prob = out.item()
            preds.append(1 if prob > 0.5 else 0)
            targets.append(target.item())

    return _render_metrics(targets, preds)


def _benchmark_standard(model, metadata: dict[str, Any], config: dict[str, Any], device: str) -> dict[str, float]:
    if not metadata["classes"]:
        raise MLXUserError("The checkpoint does not contain class labels for standard evaluation.")
    eval_dir = resolve_evaluation_dir(config["dataset_path"])
    dataset = load_standard_classification_directory(
        eval_dir,
        label_names=metadata["classes"],
        input_size=metadata["input_size"],
        colored=metadata["colored"],
    )
    console.print(f"[green]Loaded {len(dataset)} labelled images from {eval_dir}[/green]")

    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=2)
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating images"):
            images = images.to(device)
            logits = model(images)
            batch_preds = logits.argmax(dim=1).cpu().tolist()
            preds.extend(batch_preds)
            targets.extend(labels.tolist())

    return _render_metrics(targets, preds)


def _render_metrics(targets: list[int], preds: list[int]) -> dict[str, float]:
    results = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
        "f1": f1_score(targets, preds, average="macro", zero_division=0),
    }

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Score", justify="right")
    table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    table.add_row("Precision", f"{results['precision']:.4f}")
    table.add_row("Recall", f"{results['recall']:.4f}")
    table.add_row("F1-score", f"{results['f1']:.4f}")
    console.print(table)
    return results
