from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union

import typer
from rich.console import Console
from rich.table import Table

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - ensures clear guidance if dependency missing
    raise ImportError(
        "The ultralytics package (ralampay fork) is required for the obj-detect module."
    ) from exc

console = Console()


def run_obj_detect(config: Dict[str, Any]):
    action = config.get("action", "train")
    if action != "train":
        raise ValueError(f"Unsupported action '{action}' for obj-detect. Only 'train' is implemented.")

    dataset_dir = Path(config.get("dataset_path", "")).expanduser()
    if not dataset_dir.exists():
        raise typer.BadParameter(f"Dataset path does not exist: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise typer.BadParameter(f"Expected YOLO data.yaml at: {data_yaml}")

    weights_source = config.get("model_path") or config.get("model") or "yolo11n.pt"
    resolved_weights = _resolve_weights_source(weights_source)
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cpu")
    imgsz = max(config.get("height", 640), config.get("width", 640))
    project_dir = dataset_dir / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.get("run_name", "mlx-ultralytics")

    typer.secho("Ultralytics Object Detection - Training", fg=typer.colors.BRIGHT_CYAN, bold=True)

    summary = Table(title="Training Configuration", show_lines=True)
    summary.add_column("Key", justify="right", style="cyan", no_wrap=True)
    summary.add_column("Value", style="magenta")
    summary.add_row("Weights", str(resolved_weights))
    summary.add_row("Dataset", str(dataset_dir))
    summary.add_row("Data YAML", str(data_yaml))
    summary.add_row("Epochs", str(epochs))
    summary.add_row("Batch Size", str(batch_size))
    summary.add_row("Device", str(device))
    summary.add_row("Image Size", f"{imgsz}")
    summary.add_row("Project", str(project_dir))
    summary.add_row("Run Name", run_name)
    console.print(summary)

    typer.echo("Loading Ultralytics model...")
    model = YOLO(str(resolved_weights))

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "device": device,
        "batch": batch_size,
        "imgsz": imgsz,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
    }

    typer.echo("Starting training loop...")
    results = model.train(**train_kwargs)
    typer.secho("Training complete!", fg=typer.colors.GREEN, bold=True)

    return results


def _resolve_weights_source(weights_source: Union[str, Path]) -> Union[str, Path]:
    """Allow passing either weight names, local files, or ultralytics package YAML definitions."""
    if isinstance(weights_source, Path):
        return weights_source

    if not isinstance(weights_source, str):
        return weights_source

    expanded = Path(weights_source).expanduser()
    if expanded.exists():
        return expanded

    if expanded.suffix in {".yaml", ".yml"}:
        package_root = Path(ultralytics.__file__).resolve().parent
        candidates = []
        rel = expanded
        if rel.is_absolute():
            candidates.append(rel)
        else:
            candidates.extend([
                Path.cwd() / rel,
                package_root / rel,
            ])
            if rel.parts and rel.parts[0] == "ultralytics":
                stripped = Path(*rel.parts[1:])
                candidates.append(package_root / stripped)

        for candidate in candidates:
            if candidate.exists():
                return candidate

    return weights_source
