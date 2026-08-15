from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from mlx.core.exceptions import MLXUserError


MODEL_SIZES = {
    "yolo9-t": "t",
    "yolo9-s": "s",
    "yolo9-m": "m",
    "yolo9-c": "c",
}
CANONICAL_MODEL_NAMES = tuple(MODEL_SIZES)
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


def resolve_model_size(model_name: Optional[str]) -> str:
    normalized = (model_name or "").strip().lower()
    size = MODEL_SIZES.get(normalized)
    if size is None:
        available = ", ".join(CANONICAL_MODEL_NAMES)
        if not model_name:
            raise MLXUserError(
                "LibreYOLO training requires --model with a supported family and size. "
                f"Available models: {available}."
            )
        raise MLXUserError(
            f"Unsupported first-class LibreYOLO training model '{model_name}'. "
            f"Available models: {available}."
        )
    return size


def resolve_model_path(model_path: Optional[str], *, required: bool) -> Optional[Path]:
    if not model_path:
        if required:
            raise MLXUserError(
                "This LibreYOLO action requires --model-path pointing to a compatible "
                "checkpoint (.pt or .onnx)."
            )
        return None

    resolved = Path(model_path).expanduser()
    if not resolved.exists():
        raise MLXUserError(f"LibreYOLO model weights not found: {resolved}")
    if not resolved.is_file():
        raise MLXUserError(f"LibreYOLO model path is not a file: {resolved}")
    return resolved.resolve()


def resolve_dataset_source(config: dict[str, Any]) -> ResolvedDataset:
    dataset_source = str(config.get("dataset_path") or "").strip()
    if not dataset_source:
        raise MLXUserError("LibreYOLO training requires --dataset or --dataset-path.")

    output_path = config.get("output_path")
    dataset_path = Path(dataset_source).expanduser()
    if dataset_path.exists():
        if dataset_path.is_dir():
            data_yaml = dataset_path / "data.yaml"
            if not data_yaml.exists():
                raise MLXUserError(f"Expected YOLO data.yaml at: {data_yaml}")
            project_dir = Path(output_path).expanduser() if output_path else dataset_path / "runs"
            return ResolvedDataset(
                data=str(data_yaml.resolve()),
                source=str(data_yaml.resolve()),
                root_dir=dataset_path.resolve(),
                project_dir=project_dir.resolve(),
            )

        if dataset_path.suffix.lower() not in {".yaml", ".yml"}:
            raise MLXUserError(
                "LibreYOLO training expects --dataset to be a YOLO dataset directory "
                "or a .yaml/.yml dataset file."
            )
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

    normalized = DATASET_ALIASES.get(dataset_source.lower(), dataset_source)
    if normalized == dataset_source and Path(dataset_source).suffix.lower() in {".yaml", ".yml"}:
        raise MLXUserError(f"LibreYOLO dataset YAML not found: {dataset_path}")

    project_dir = (
        Path(output_path).expanduser()
        if output_path
        else Path.cwd() / "runs" / "object_detection"
    )
    return ResolvedDataset(
        data=normalized,
        source=normalized,
        root_dir=None,
        project_dir=project_dir.resolve(),
    )


def dependency_error(action: str) -> MLXUserError:
    return MLXUserError(
        "The LibreYOLO provider is not installed. Install it from the configured "
        "Ralampay fork with 'pip install \".[object-detection-libreyolo]\"' "
        f"before {action}."
    )
