from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.artifacts import resolve_onnx_output_target
from mlx.modes.object_detection.requests import ConvertObjectDetectionRequest
from mlx.modes.object_detection.ultralytics.utils import resolve_imgsz, resolve_model_paths

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ConvertUltralyticsObjectDetectionModel:
    def __init__(
        self,
        config: dict[str, Any] | ConvertObjectDetectionRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.config = (
            config.to_config()
            if isinstance(config, ConvertObjectDetectionRequest)
            else dict(config)
        )
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> Path:
        return _run_conversion(self.config, reporter=self.reporter)


def convert_object_detection_model(config: dict[str, Any]) -> Path:
    """Compatibility wrapper around the Ultralytics conversion command."""

    from mlx.modes.object_detection.presentation import RichWorkflowReporter

    return ConvertUltralyticsObjectDetectionModel(
        config,
        reporter=RichWorkflowReporter(),
    ).execute()


def _run_conversion(
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> Path:
    reporter = reporter or NullWorkflowReporter()
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

    emit(
        reporter,
        "info",
        "Ultralytics object-detection ONNX export configured.",
        payload={
            "event": "conversion_summary",
            "values": {
                "Model Path": resolved_weights,
                "Output Path": output_target,
                "Device": str(config.get("device", "cpu")),
                "Image Size": imgsz,
            },
        },
    )

    emit(reporter, "info", "Loading Ultralytics checkpoint...")
    model = YOLO(str(resolved_weights))
    emit(reporter, "info", "Exporting checkpoint to ONNX...")
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

    emit(reporter, "success", f"ONNX export complete: {final_path}")
    return final_path
