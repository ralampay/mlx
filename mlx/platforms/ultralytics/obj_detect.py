from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

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
        return _run_stream_inference(config, source="camera")
    if action == "infer-video":
        return _run_stream_inference(config, source="video")
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

    palette = _get_color_palette(names)

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names.get(int(cls), str(int(cls)))
        text = f"{label}: {conf:.2f}"
        color = palette.get(label, palette.get(int(cls), (0, 255, 0)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def _resolve_model_paths(
    config: Dict[str, Any],
    require_yaml: bool,
    require_weights: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    model_cfg = config.get("model")
    resolved_cfg = Path(_resolve_weights_source(model_cfg)) if model_cfg else None
    if require_yaml and resolved_cfg is None:
        raise typer.BadParameter("This action requires --model pointing to the model YAML.")
    if resolved_cfg and not resolved_cfg.exists():
        raise typer.BadParameter(f"Model YAML not found: {resolved_cfg}")

    weights_path = config.get("model_path")
    resolved_weights = Path(_resolve_weights_source(weights_path)) if weights_path else None
    if require_weights and resolved_weights is None:
        raise typer.BadParameter("This action requires --model-path pointing to trained weights (.pt).")
    if resolved_weights and not resolved_weights.exists():
        raise typer.BadParameter(f"Model weights not found: {resolved_weights}")

    return resolved_cfg, resolved_weights


def _initialize_model(
    resolved_cfg: Optional[Path],
    resolved_weights: Optional[Path],
    prefer_cfg: bool,
) -> YOLO:
    model: Optional[YOLO] = None

    if prefer_cfg and resolved_cfg:
        model = YOLO(str(resolved_cfg))

    if resolved_weights:
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

    if model is None and resolved_cfg:
        model = YOLO(str(resolved_cfg))

    if model is None:
        raise RuntimeError("Failed to initialize the YOLO model. Check --model and --model-path arguments.")

    return model


def _get_color_palette(names: Dict[int, str]):
    import hashlib

    cache = getattr(_get_color_palette, "_cache", None)
    if cache is None:
        cache = {}
        _get_color_palette._cache = cache

    def color_for_label(label: str):
        if label in cache:
            return cache[label]
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
        r = int(digest[0:2], 16)
        g = int(digest[2:4], 16)
        b = int(digest[4:6], 16)
        base = (r, g, b)
        cache[label] = tuple(int(min(max(c, 64), 255)) for c in base)
        return cache[label]

    palette = {}
    for idx, label in names.items():
        palette[label] = color_for_label(str(label))
        palette[idx] = palette[label]
    return palette


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
