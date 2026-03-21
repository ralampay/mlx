from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from mlx.core.exceptions import MLXUserError

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for object-detection inference. Install it with 'pip install opencv-python'."
    ) from exc

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "The ultralytics package (ralampay fork) is required for the obj-detect module."
    ) from exc


def resolve_weights_source(weights_source: Union[str, Path, None]) -> Union[str, Path, None]:
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
        if expanded.is_absolute():
            candidates.append(expanded)
        else:
            candidates.extend((Path.cwd() / expanded, package_root / expanded))
            if expanded.parts and expanded.parts[0] == "ultralytics":
                candidates.append(package_root / Path(*expanded.parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return weights_source


def resolve_model_paths(
    config: dict[str, Any],
    *,
    require_yaml: bool,
    require_weights: bool,
) -> tuple[Optional[Path], Optional[Path]]:
    model_cfg = config.get("model")
    resolved_cfg = Path(resolve_weights_source(model_cfg)) if model_cfg else None
    if require_yaml and resolved_cfg is None:
        raise MLXUserError("This action requires --model pointing to the model YAML.")
    if resolved_cfg and not resolved_cfg.exists():
        raise MLXUserError(f"Model YAML not found: {resolved_cfg}")

    weights_path = config.get("model_path")
    resolved_weights = Path(resolve_weights_source(weights_path)) if weights_path else None
    if require_weights and resolved_weights is None:
        raise MLXUserError("This action requires --model-path pointing to trained weights (.pt).")
    if resolved_weights and not resolved_weights.exists():
        raise MLXUserError(f"Model weights not found: {resolved_weights}")

    return resolved_cfg, resolved_weights


def initialize_model(
    resolved_cfg: Optional[Path],
    resolved_weights: Optional[Path],
    *,
    prefer_cfg: bool,
) -> YOLO:
    model: Optional[YOLO] = None

    if prefer_cfg and resolved_cfg:
        model = YOLO(str(resolved_cfg))

    if resolved_weights:
        if model is None:
            model = YOLO(str(resolved_weights))
        else:
            load = getattr(model, "load", None)
            if callable(load):
                loaded = model.load(str(resolved_weights))
                if loaded is not None:
                    model = loaded
            else:
                model = YOLO(str(resolved_weights))

    if model is None and resolved_cfg:
        model = YOLO(str(resolved_cfg))
    if model is None:
        raise RuntimeError("Failed to initialize the YOLO model. Check --model and --model-path.")
    return model


def annotate_detections(frame, result):
    annotated = frame.copy()
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return annotated

    names = result.names or {}
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    classes = (
        boxes.cls.cpu().numpy().astype(int)
        if boxes.cls is not None
        else np.zeros(len(xyxy), dtype=int)
    )

    palette = _color_palette(names)
    for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = names.get(int(class_id), str(int(class_id)))
        text = f"{label}: {confidence:.2f}"
        color = palette.get(label, palette.get(int(class_id), (0, 255, 0)))
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


def _color_palette(names: dict[int, str]) -> dict[Any, tuple[int, int, int]]:
    cache = getattr(_color_palette, "_cache", None)
    if cache is None:
        cache = {}
        _color_palette._cache = cache

    palette = {}
    for idx, label in names.items():
        palette[label] = _color_for_label(str(label), cache)
        palette[idx] = palette[label]
    return palette


def _color_for_label(label: str, cache: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    if label in cache:
        return cache[label]
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    color = tuple(int(min(max(int(digest[i : i + 2], 16), 64), 255)) for i in (0, 2, 4))
    cache[label] = color
    return color
