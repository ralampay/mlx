from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_success
from mlx.modes.object_detection.artifacts import resolve_onnx_output_target
from mlx.modes.object_detection.ultralytics.utils import resolve_imgsz, resolve_model_paths

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ConvertUltralyticsObjectDetectionModel:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def execute(self) -> Path:
        return _run_conversion(self.config)


def convert_object_detection_model(config: dict[str, Any]) -> Path:
    """Compatibility wrapper around the Ultralytics conversion command."""

    return ConvertUltralyticsObjectDetectionModel(config).execute()


def _run_conversion(config: dict[str, Any]) -> Path:
    if YOLO is None:
        raise MLXUserError(
            "The Ultralytics provider is not installed. Install the detection dependencies before exporting."
        )
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

    output_target = resolve_onnx_output_target(config, resolved_weights)
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
