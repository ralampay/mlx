from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - ensures clear guidance if dependency missing
    raise ImportError(
        "The ultralytics package (ralampay fork) is required for the obj-detect module."
    ) from exc
import hashlib

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for --action infer-camera. Install it with 'pip install opencv-python'."
    ) from exc

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

def _get_color_palette(names: Dict[int, str]):
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

def _annotate_detections(frame, result):
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
