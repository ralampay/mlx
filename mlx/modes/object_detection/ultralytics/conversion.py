from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_success
from mlx.modes.object_detection.ultralytics.utils import resolve_imgsz, resolve_model_paths

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "The ultralytics package (ralampay fork) is required for object-detection mode."
    ) from exc


def convert_object_detection_model(config: dict[str, Any]) -> Path:
    _, resolved_weights = resolve_model_paths(
        config,
        require_yaml=False,
        require_weights=True,
    )
    if resolved_weights is None:
        raise MLXUserError("This action requires --model-path pointing to trained weights (.pt).")
    if resolved_weights.suffix.lower() != ".pt":
        raise MLXUserError(
            "The convert action expects --model-path to point to an Ultralytics PyTorch checkpoint (.pt)."
        )

    output_target = _resolve_output_target(config, resolved_weights)
    output_target.parent.mkdir(parents=True, exist_ok=True)
    imgsz = resolve_imgsz(config)

    console.print(Panel.fit("Ultralytics Object Detection - ONNX Export", border_style="cyan"))
    console.print(
        _conversion_summary_table(
            model_path=resolved_weights,
            output_path=output_target,
            device=str(config.get("device", "cpu")),
            imgsz=imgsz,
        )
    )

    print_info("Loading Ultralytics checkpoint...")
    model = YOLO(str(resolved_weights))
    print_info("Exporting checkpoint to ONNX...")
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            device=config.get("device", "cpu"),
        )
    ).resolve()

    final_path = exported_path
    if exported_path != output_target.resolve():
        exported_path.replace(output_target)
        final_path = output_target.resolve()

    print_success(f"ONNX export complete: {final_path}")
    return final_path


def _resolve_output_target(config: dict[str, Any], resolved_weights: Path) -> Path:
    output_path = config.get("output_path")
    default_target = resolved_weights.with_suffix(".onnx")
    if not output_path:
        return default_target.resolve()

    candidate = Path(output_path).expanduser()
    if candidate.exists() and candidate.is_dir():
        return (candidate / default_target.name).resolve()
    if candidate.suffix.lower() == ".onnx":
        return candidate.resolve()
    if candidate.exists():
        return (candidate / default_target.name).resolve()
    if candidate.suffix:
        raise MLXUserError("For ONNX export, --output must be a directory or a path ending in .onnx.")
    return (candidate / default_target.name).resolve()


def _conversion_summary_table(
    *,
    model_path: Path,
    output_path: Path,
    device: str,
    imgsz,
) -> Table:
    summary = Table(title="Conversion Configuration", show_lines=True)
    summary.add_column("Key", justify="right", style="cyan", no_wrap=True)
    summary.add_column("Value", style="magenta")
    summary.add_row("Source Checkpoint", str(model_path))
    summary.add_row("Export Format", "onnx")
    summary.add_row("Output Path", str(output_path))
    summary.add_row("Device", device)
    summary.add_row("Image Size", str(imgsz))
    return summary
