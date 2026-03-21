from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from rich.panel import Panel
from rich.table import Table

from mlx.definitions import MLXUserError
from .utils import _resolve_model_paths, _initialize_model
from .run_stream_inference import RunStreamInference
from mlx.ui import console, print_info, print_success

def run_obj_detect(config: Dict[str, Any]):
    action = config.get("action", "train")

    if action == "train":
        return _train_obj_detect(config)
    if action == "infer-camera":
        #return _run_stream_inference(config, source="camera")
        cmd = RunStreamInference(
            config,
            source="camera"
        )

        return cmd.execute()
    if action == "infer-video":
        cmd = RunStreamInference(
            config,
            source="video"
        )

        return cmd.execute()
    raise ValueError(
        f"Unsupported action '{action}' for obj-detect. Supported actions: train, infer-camera, infer-video."
    )


def _train_obj_detect(config: Dict[str, Any]):
    dataset_dir = Path(config.get("dataset_path", "")).expanduser()
    if not dataset_dir.exists():
        raise MLXUserError(f"Dataset path does not exist: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise MLXUserError(f"Expected YOLO data.yaml at: {data_yaml}")

    resolved_cfg, resolved_weights = _resolve_model_paths(
        config, require_yaml=True, require_weights=False
    )
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cpu")
    imgsz = max(config.get("height", 640), config.get("width", 640))
    project_dir = dataset_dir / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.get("run_name", "mlx-ultralytics")

    console.print(Panel.fit("Ultralytics Object Detection - Training", border_style="cyan"))

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
    summary.add_row("Image Size", f"{imgsz}")
    summary.add_row("Project", str(project_dir))
    summary.add_row("Run Name", run_name)
    summary.add_row("Pretrained", str(bool(config.get("pretrained", False))))
    lr0 = config.get("lr0")
    summary.add_row("lr0", str(lr0) if lr0 is not None else "default")
    summary.add_row("Optimizer", config.get("optimizer", "auto"))
    summary.add_row("nbs", str(config.get("nbs", 64)))
    summary.add_row("Warmup Epochs", str(config.get("warmup_epochs", 3.0)))
    summary.add_row("AMP", str(bool(config.get("amp", True))))
    loss_clip = config.get("loss_clip")
    summary.add_row("Loss Clip", str(loss_clip) if loss_clip is not None else "disabled")
    console.print(summary)

    print_info("Loading Ultralytics model...")
    model = _initialize_model(resolved_cfg, resolved_weights, prefer_cfg=True)

    overrides = getattr(model, "overrides", {})
    overrides["pretrained"] = bool(config.get("pretrained", False))
    overrides["model"] = str(resolved_cfg) if resolved_cfg else overrides.get("model")
    overrides.pop("weights", None)
    overrides["optimizer"] = config.get("optimizer", overrides.get("optimizer", "auto"))
    overrides["nbs"] = int(config.get("nbs", overrides.get("nbs", 64)))
    overrides["warmup_epochs"] = float(config.get("warmup_epochs", overrides.get("warmup_epochs", 3.0)))
    overrides["amp"] = bool(config.get("amp", overrides.get("amp", True)))
    model.overrides = overrides
    model.ckpt_path = str(resolved_weights) if resolved_weights else None

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "device": device,
        "batch": batch_size,
        "imgsz": imgsz,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": overrides["pretrained"],
    }
    if lr0 is not None:
        train_kwargs["lr0"] = float(lr0)
    loss_clip = config.get("loss_clip")
    if loss_clip is not None:
        train_kwargs["loss_clip"] = float(loss_clip)

    print_info("Starting training loop...")
    results = model.train(**train_kwargs)
    print_success("Training complete!")

    return results
