from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True)
class LibreYOLOModelSpec:
    size: str
    drax_stages: tuple[str, ...] = ()

    @property
    def uses_drax(self) -> bool:
        return bool(self.drax_stages)


MODEL_SPECS = {
    "yolo9-t": LibreYOLOModelSpec(size="t"),
    "yolo9-s": LibreYOLOModelSpec(size="s"),
    "yolo9-m": LibreYOLOModelSpec(size="m"),
    "yolo9-c": LibreYOLOModelSpec(size="c"),
    "yolo9-s-drax-b5": LibreYOLOModelSpec(size="s", drax_stages=("b5",)),
}
CANONICAL_MODEL_NAMES = tuple(MODEL_SPECS)
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


def resolve_model_spec(model_name: Optional[str]) -> LibreYOLOModelSpec:
    normalized = (model_name or "").strip().lower()
    model_spec = MODEL_SPECS.get(normalized)
    if model_spec is None:
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
    return model_spec


def resolve_model_size(model_name: Optional[str]) -> str:
    return resolve_model_spec(model_name).size


def build_drax_config(model_spec: LibreYOLOModelSpec) -> Any:
    if not model_spec.uses_drax:
        return None

    try:
        from libreyolo.models.yolo9 import DraxConfig
    except ImportError as exc:
        raise MLXUserError(
            "The installed LibreYOLO fork does not expose the required DraxConfig API. "
            "Run './update.sh' from the MLX repository and try again."
        ) from exc

    return DraxConfig(
        enabled=True,
        stages=model_spec.drax_stages,
        use_attention=True,
        efficient=True,
        fusion_mode="average",
        drop_path=0.0,
    )


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
