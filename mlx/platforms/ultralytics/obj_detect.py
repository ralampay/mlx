from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import typer
from rich.console import Console
from rich.table import Table

from .utils import _resolve_weights_source, _resolve_model_paths, _get_color_palette
from .run_stream_inference import RunStreamInference

console = Console()

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
        raise typer.BadParameter(f"Dataset path does not exist: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise typer.BadParameter(f"Expected YOLO data.yaml at: {data_yaml}")

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

    typer.secho("Ultralytics Object Detection - Training", fg=typer.colors.BRIGHT_CYAN, bold=True)

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

    typer.echo("Loading Ultralytics model...")
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

    typer.echo("Starting training loop...")
    results = model.train(**train_kwargs)
    typer.secho("Training complete!", fg=typer.colors.GREEN, bold=True)

    return results


def _run_stream_inference(config: Dict[str, Any], source: str):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "OpenCV is required for --action infer-camera. Install it with 'pip install opencv-python'."
        ) from exc

    resolved_cfg, resolved_weights = _resolve_model_paths(
        config, require_yaml=True, require_weights=True
    )

    device = config.get("device", "cpu")
    imgsz = max(config.get("height", 640), config.get("width", 640))
    confidence = float(config.get("confidence", 0.25))
    camera_index = int(config.get("camera_index", 0))
    if source == "camera":
        typer.secho("Ultralytics Object Detection - Camera Inference", fg=typer.colors.BRIGHT_CYAN, bold=True)
    else:
        typer.secho("Ultralytics Object Detection - Video Inference", fg=typer.colors.BRIGHT_CYAN, bold=True)

    if resolved_cfg:
        typer.echo(f"Model YAML: {resolved_cfg}")
    typer.echo(f"Loading weights from: {resolved_weights}")

    model = _initialize_model(resolved_cfg, resolved_weights, prefer_cfg=False)

    typer.echo(f"Using device: {device} | Image size: {imgsz} | Confidence: {confidence}")
    typer.secho("Press 'q' or 'Esc' to exit.", fg=typer.colors.YELLOW)

    if source == "camera":
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}.")
        window_title = "MLX Object Detection (Camera)"
    elif source == "video":
        video_path = config.get("file_path")
        if not video_path:
            raise typer.BadParameter("Video inference requires --file-path pointing to the video file.")
        resolved_video = Path(video_path).expanduser()
        if not resolved_video.exists():
            raise typer.BadParameter(f"Video file not found: {resolved_video}")
        cap = cv2.VideoCapture(str(resolved_video))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {resolved_video}")
        window_title = f"MLX Object Detection (Video: {resolved_video.name})"
    else:
        raise ValueError(f"Unsupported source type: {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                typer.echo("No more frames to process." if source == "video" else "Failed to read frame from camera.")
                break

            result = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=confidence,
                device=device,
                verbose=False,
                stream=False,
            )

            annotated = _annotate_detections(frame, result[0])
            cv2.imshow(window_title, annotated)

            key = cv2.waitKey(1 if source == "camera" else 10) & 0xFF
            if key in (ord("q"), 27):
                typer.echo("Exiting inference.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()




