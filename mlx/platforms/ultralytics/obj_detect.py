from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Union

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
    if action == "train":
        return _train_obj_detect(config)
    if action == "infer-camera":
        return _infer_camera(config)
    raise ValueError(f"Unsupported action '{action}' for obj-detect. Supported actions: train, infer-camera.")


def _train_obj_detect(config: Dict[str, Any]):
    dataset_dir = Path(config.get("dataset_path", "")).expanduser()
    if not dataset_dir.exists():
        raise typer.BadParameter(f"Dataset path does not exist: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise typer.BadParameter(f"Expected YOLO data.yaml at: {data_yaml}")

    weights_source = config.get("model_path")
    resolved_weights = _resolve_weights_source(weights_source) if weights_source else None
    if isinstance(resolved_weights, (str, Path)):
        resolved_weights = Path(resolved_weights)
        if not resolved_weights.exists():
            raise typer.BadParameter(f"Model weights not found: {resolved_weights}")
    model_config = config.get("model")
    resolved_cfg = Path(_resolve_weights_source(model_config)) if model_config else None
    if resolved_cfg and not resolved_cfg.exists():
        raise typer.BadParameter(f"Model YAML not found: {resolved_cfg}")
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
    if not model_config and not resolved_weights:
        raise typer.BadParameter("Provide --model (YAML) and/or --model-path (.pt) to define the detector.")

    model: Optional[YOLO] = None

    if resolved_cfg:
        model = YOLO(str(resolved_cfg))

    if resolved_weights:
        typer.echo(f"Loading weights from: {resolved_weights}")
        if model is None:
            model = YOLO(str(resolved_weights))
        else:
            load_result = getattr(model, "load", None)
            if callable(load_result):
                loaded = model.load(str(resolved_weights))
                if loaded is not None:
                    model = loaded
            else:
                model = YOLO(str(resolved_weights))

    if model is None:
        raise RuntimeError("Failed to initialize the YOLO model. Check --model and --model-path arguments.")

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


def _infer_camera(config: Dict[str, Any]):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "OpenCV is required for --action infer-camera. Install it with 'pip install opencv-python'."
        ) from exc

    weights_path = config.get("model_path")
    if not weights_path:
        raise typer.BadParameter("Camera inference requires --model-path pointing to trained YOLO weights (.pt).")

    resolved_weights = Path(weights_path).expanduser()
    if not resolved_weights.exists():
        raise typer.BadParameter(f"Model weights not found: {resolved_weights}")

    model_cfg = config.get("model")
    if not model_cfg:
        raise typer.BadParameter("Camera inference requires --model pointing to the YOLO model YAML.")

    resolved_cfg = Path(_resolve_weights_source(model_cfg))
    if not resolved_cfg.exists():
        raise typer.BadParameter(f"Model YAML not found: {resolved_cfg}")

    device = config.get("device", "cpu")
    imgsz = max(config.get("height", 640), config.get("width", 640))
    confidence = float(config.get("confidence", 0.25))
    camera_index = int(config.get("camera_index", 0))

    typer.secho("Ultralytics Object Detection - Camera Inference", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.echo(f"Loading model architecture from: {resolved_cfg}")
    model = YOLO(str(resolved_cfg))

    typer.echo(f"Loading weights from: {resolved_weights}")
    load_result = getattr(model, "load", None)
    if callable(load_result):
        loaded = model.load(str(resolved_weights))
        if loaded is not None:
            model = loaded
    else:  # fallback: re-instantiate straight from weights
        model = YOLO(str(resolved_weights))

    typer.echo(f"Using device: {device} | Image size: {imgsz} | Confidence: {confidence}")
    typer.secho("Press 'q' or 'Esc' to exit the camera feed.", fg=typer.colors.YELLOW)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    window_title = "MLX Object Detection"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                typer.echo("Failed to read frame from camera. Exiting.")
                break

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=confidence,
                device=device,
                verbose=False,
                stream=False,
            )

            annotated = _annotate_detections(frame, results[0])
            cv2.imshow(window_title, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                typer.echo("Exiting camera inference.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _annotate_detections(frame, result):
    import cv2
    import numpy as np

    annotated = frame.copy()
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return annotated

    names = result.names or {}
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names.get(int(cls), str(int(cls)))
        text = f"{label}: {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def _resolve_weights_source(weights_source: Union[str, Path, None]) -> Union[str, Path, None]:
    """Allow passing either weight names, local files, or ultralytics package YAML definitions."""
    if weights_source is None:
        return None

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
