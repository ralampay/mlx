from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_success
from mlx.modes.object_detection.ultralytics.utils import initialize_model, resolve_model_paths


def train_object_detection(config: dict[str, Any]):
    dataset_dir = Path(config.get("dataset_path", "")).expanduser()
    if not dataset_dir.exists():
        raise MLXUserError(f"Dataset path does not exist: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise MLXUserError(f"Expected YOLO data.yaml at: {data_yaml}")

    resolved_cfg, resolved_weights = resolve_model_paths(
        config,
        require_yaml=True,
        require_weights=False,
    )
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cpu")
    imgsz = max(config.get("height", 640), config.get("width", 640))
    project_dir = dataset_dir / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.get("run_name", "mlx-ultralytics")
    lr0 = config.get("lr0")
    loss_clip = config.get("loss_clip")

    console.print(Panel.fit("Ultralytics Object Detection - Training", border_style="cyan"))
    console.print(_training_summary_table(
        resolved_cfg=resolved_cfg,
        resolved_weights=resolved_weights,
        dataset_dir=dataset_dir,
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        imgsz=imgsz,
        project_dir=project_dir,
        run_name=run_name,
        config=config,
    ))

    print_info("Loading Ultralytics model...")
    model = initialize_model(resolved_cfg, resolved_weights, prefer_cfg=True)
    overrides = getattr(model, "overrides", {})
    overrides["pretrained"] = bool(config.get("pretrained", False))
    overrides["model"] = str(resolved_cfg) if resolved_cfg else overrides.get("model")
    overrides.pop("weights", None)
    overrides["optimizer"] = config.get("optimizer", overrides.get("optimizer", "auto"))
    overrides["nbs"] = int(config.get("nbs", overrides.get("nbs", 64)))
    overrides["warmup_epochs"] = float(
        config.get("warmup_epochs", overrides.get("warmup_epochs", 3.0))
    )
    overrides["amp"] = bool(config.get("amp", overrides.get("amp", True)))
    model.overrides = overrides
    model.ckpt_path = str(resolved_weights) if resolved_weights else None

    train_kwargs = {
        "batch": batch_size,
        "data": str(data_yaml),
        "device": device,
        "epochs": epochs,
        "exist_ok": True,
        "imgsz": imgsz,
        "name": run_name,
        "pretrained": overrides["pretrained"],
        "project": str(project_dir),
    }
    if lr0 is not None:
        train_kwargs["lr0"] = float(lr0)
    if loss_clip is not None:
        train_kwargs["loss_clip"] = float(loss_clip)
    if config.get("random_seed") is not None:
        train_kwargs["seed"] = int(config["random_seed"])

    print_info("Starting training loop...")
    results = model.train(**train_kwargs)
    print_success("Training complete!")
    return results


def _training_summary_table(
    *,
    resolved_cfg,
    resolved_weights,
    dataset_dir: Path,
    data_yaml: Path,
    epochs: int,
    batch_size: int,
    device: str,
    imgsz: int,
    project_dir: Path,
    run_name: str,
    config: dict[str, Any],
) -> Table:
    summary = Table(title="Training Configuration", show_lines=True)
    summary.add_column("Key", justify="right", style="cyan", no_wrap=True)
    summary.add_column("Value", style="magenta")
    summary.add_row("Init Weights", str(resolved_weights) if resolved_weights else "random init")
    summary.add_row("Model YAML", str(resolved_cfg) if resolved_cfg else "not set")
    summary.add_row("Dataset", str(dataset_dir))
    summary.add_row("Data YAML", str(data_yaml))
    summary.add_row("Epochs", str(epochs))
    summary.add_row("Batch Size", str(batch_size))
    summary.add_row("Device", str(device))
    summary.add_row("Image Size", str(imgsz))
    summary.add_row("Project", str(project_dir))
    summary.add_row("Run Name", run_name)
    summary.add_row("Pretrained", str(bool(config.get("pretrained", False))))
    summary.add_row("lr0", str(config.get("lr0")) if config.get("lr0") is not None else "default")
    summary.add_row("Optimizer", config.get("optimizer", "auto"))
    summary.add_row("nbs", str(config.get("nbs", 64)))
    summary.add_row("Warmup Epochs", str(config.get("warmup_epochs", 3.0)))
    summary.add_row("AMP", str(bool(config.get("amp", True))))
    summary.add_row(
        "Loss Clip",
        str(config.get("loss_clip")) if config.get("loss_clip") is not None else "disabled",
    )
    return summary
