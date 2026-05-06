from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
        "The ultralytics package (ralampay fork) is required for object-detection mode."
    ) from exc


MODEL_ALIASES = {
    "yolo26": "yolo26.yaml",
    "yolo26.yaml": "yolo26.yaml",
    "yolov26": "yolo26.yaml",
    "yolov26.yaml": "yolo26.yaml",
    "draxnet-yolo26": "draxnet-yolo26.yaml",
    "draxnet-yolo26.yaml": "draxnet-yolo26.yaml",
}

DATASET_ALIASES = {
    "coco8": "coco8.yaml",
    "coco8.yaml": "coco8.yaml",
    "coco128": "coco128.yaml",
    "coco128.yaml": "coco128.yaml",
}


@dataclass(frozen=True)
class ResolvedDataset:
    data: str
    source: str
    root_dir: Optional[Path]
    project_dir: Path


def resolve_imgsz(config: dict[str, Any]) -> Union[int, tuple[int, int]]:
    height = int(config.get("height", 640))
    width = int(config.get("width", 640))
    return height if height == width else (height, width)


def _ultralytics_package_root() -> Path:
    return Path(ultralytics.__file__).resolve().parent


def _ultralytics_models_root() -> Path:
    return _ultralytics_package_root() / "cfg" / "models"


def _resolve_with_candidates(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_yaml_in_package(
    source: str,
    *,
    package_subdir: str,
    aliases: dict[str, str],
) -> Optional[Path]:
    package_root = _ultralytics_package_root()
    normalized = aliases.get(source.lower(), source)
    candidates = []
    expanded = Path(normalized).expanduser()

    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        candidates.extend((Path.cwd() / expanded, package_root / expanded))
        if expanded.parts and expanded.parts[0] == "ultralytics":
            candidates.append(package_root / Path(*expanded.parts[1:]))
        package_dir = package_root / package_subdir
        candidates.append(package_dir / expanded.name)
        if expanded.suffix:
            candidates.extend(package_dir.rglob(expanded.name))
        else:
            candidates.extend(package_dir.rglob(f"{expanded.name}.yaml"))
            candidates.extend(package_dir.rglob(f"{expanded.name}.yml"))

    return _resolve_with_candidates(candidates)


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
        resolved_yaml = _resolve_yaml_in_package(
            weights_source,
            package_subdir="cfg/models",
            aliases=MODEL_ALIASES,
        )
        if resolved_yaml is not None:
            return resolved_yaml

    resolved_model = _resolve_yaml_in_package(
        weights_source,
        package_subdir="cfg/models",
        aliases=MODEL_ALIASES,
    )
    if resolved_model is not None:
        return resolved_model

    return weights_source


def resolve_model_paths(
    config: dict[str, Any],
    *,
    require_yaml: bool,
    require_weights: bool,
) -> tuple[Optional[Path], Optional[Path]]:
    model_cfg = config.get("model")
    resolved_cfg = Path(resolve_weights_source(model_cfg)) if model_cfg else None
    weights_path = config.get("model_path")
    resolved_weights = Path(resolve_weights_source(weights_path)) if weights_path else None
    weights_suffix = resolved_weights.suffix.lower() if resolved_weights else None
    requires_cfg_for_runtime = require_yaml and weights_suffix != ".onnx"

    if requires_cfg_for_runtime and resolved_cfg is None:
        raise MLXUserError("This action requires --model pointing to the model YAML.")
    if resolved_cfg and not resolved_cfg.exists():
        raise MLXUserError(_build_missing_model_yaml_message(model_cfg, resolved_cfg))
    if require_weights and resolved_weights is None:
        raise MLXUserError("This action requires --model-path pointing to trained weights (.pt).")
    if resolved_weights and not resolved_weights.exists():
        raise MLXUserError(f"Model weights not found: {resolved_weights}")

    return resolved_cfg, resolved_weights


def _build_missing_model_yaml_message(model_cfg: Any, resolved_cfg: Path) -> str:
    requested = str(model_cfg)
    normalized = MODEL_ALIASES.get(requested.lower(), requested)
    models_root = _ultralytics_models_root()

    if normalized == "draxnet-yolo26.yaml":
        yolo26_yaml = _resolve_yaml_in_package(
            "yolo26.yaml",
            package_subdir="cfg/models",
            aliases=MODEL_ALIASES,
        )
        if yolo26_yaml is not None:
            return (
                "Model YAML not found: draxnet-yolo26. "
                f"The alias resolves to `{normalized}`, but that file is not present under "
                f"`{models_root}` in the installed `ultralytics` package. "
                "Your current environment has `yolo26.yaml`, but not the custom DraxNet variant. "
                "Reinstall the pinned `ultralytics` dependency for this repo, use a direct path to "
                "`draxnet-yolo26.yaml`, or switch to `--model yolo26`."
            )

    if requested.lower() in MODEL_ALIASES:
        return (
            f"Model YAML not found: {requested}. "
            f"The alias resolves to `{normalized}`, but that file is not present under `{models_root}`."
        )

    return f"Model YAML not found: {resolved_cfg}"


def resolve_dataset_source(config: dict[str, Any]) -> ResolvedDataset:
    dataset_source = config.get("dataset_path", "")
    if not dataset_source:
        raise MLXUserError("This action requires --dataset or --dataset-path.")

    output_path = config.get("output_path")
    dataset_path = Path(dataset_source).expanduser()
    if dataset_path.exists():
        if dataset_path.is_dir():
            data_yaml = dataset_path / "data.yaml"
            if not data_yaml.exists():
                raise MLXUserError(f"Expected YOLO data.yaml at: {data_yaml}")
            project_dir = (
                Path(output_path).expanduser()
                if output_path
                else dataset_path / "runs"
            )
            return ResolvedDataset(
                data=str(data_yaml),
                source=str(data_yaml),
                root_dir=dataset_path.resolve(),
                project_dir=project_dir.resolve(),
            )

        if dataset_path.suffix in {".yaml", ".yml"}:
            project_dir = (
                Path(output_path).expanduser()
                if output_path
                else Path.cwd() / "runs" / "object_detection"
            )
            return ResolvedDataset(
                data=str(dataset_path.resolve()),
                source=str(dataset_path.resolve()),
                root_dir=None,
                project_dir=project_dir.resolve(),
            )

    resolved_dataset_yaml = _resolve_yaml_in_package(
        dataset_source,
        package_subdir="cfg/datasets",
        aliases=DATASET_ALIASES,
    )
    if resolved_dataset_yaml is None:
        raise MLXUserError(
            "Dataset source not found. Pass a YOLO dataset directory containing data.yaml, "
            "a dataset YAML path, or a built-in alias such as 'coco8' or 'coco128'."
        )

    project_dir = (
        Path(output_path).expanduser()
        if output_path
        else Path.cwd() / "runs" / "object_detection"
    )
    return ResolvedDataset(
        data=str(resolved_dataset_yaml),
        source=str(resolved_dataset_yaml),
        root_dir=None,
        project_dir=project_dir.resolve(),
    )


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
    if result is None:
        return annotated

    if hasattr(result, "detections"):
        names = getattr(result, "names", {}) or {}
        detections = getattr(result, "detections", []) or []
    else:
        if result.boxes is None or len(result.boxes) == 0:
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
        detections = [
            {
                "xyxy": tuple(map(int, coords)),
                "confidence": float(confidence),
                "class_id": int(class_id),
                "label": names.get(int(class_id), str(int(class_id))),
            }
            for coords, confidence, class_id in zip(xyxy, confs, classes)
        ]

    if not detections:
        return annotated

    palette = _color_palette(names)
    for detection in detections:
        if hasattr(detection, "xyxy"):
            x1, y1, x2, y2 = detection.xyxy
            confidence = detection.confidence
            class_id = detection.class_id
            label = detection.label
        else:
            x1, y1, x2, y2 = detection["xyxy"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]
            label = detection["label"]
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
