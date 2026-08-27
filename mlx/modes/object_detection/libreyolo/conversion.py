from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.artifacts import resolve_onnx_output_target
from mlx.modes.object_detection.libreyolo.utils import (
    dependency_error,
    resolve_imgsz,
    resolve_model_path,
)
from mlx.modes.object_detection.requests import ConvertObjectDetectionRequest


class ConvertLibreYOLOObjectDetectionModel:
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
        try:
            from libreyolo import LibreYOLO
        except ImportError as exc:
            raise dependency_error("exporting an object-detection model") from exc

        resolved_weights = resolve_model_path(self.config.get("model_path"), required=True)
        if resolved_weights.suffix.lower() != ".pt":
            raise MLXUserError(
                "LibreYOLO conversion requires --model-path to point to a PyTorch checkpoint (.pt)."
            )

        output_target = resolve_onnx_output_target(self.config, resolved_weights)
        output_target.parent.mkdir(parents=True, exist_ok=True)
        device = str(self.config.get("device", "cpu"))
        image_size = resolve_imgsz(self.config)

        emit(
            self.reporter,
            "info",
            "LibreYOLO object-detection ONNX export configured.",
            payload={
                "event": "conversion_summary",
                "values": {
                    "Model Path": resolved_weights,
                    "Output Path": output_target,
                    "Device": device,
                    "Image Size": image_size,
                },
            },
        )

        try:
            emit(self.reporter, "info", f"Loading LibreYOLO checkpoint: {resolved_weights}")
            model = LibreYOLO(str(resolved_weights), device=device, task="detect")
            emit(self.reporter, "info", "Exporting LibreYOLO checkpoint to ONNX...")
            exported_path = Path(
                model.export(format="onnx", imgsz=image_size, device=device)
            ).expanduser()
            if not exported_path.exists():
                raise MLXUserError(
                    f"LibreYOLO reported an export path that does not exist: {exported_path}"
                )
            exported_path = exported_path.resolve()
            final_path = output_target.resolve()
            if exported_path != final_path:
                exported_path.replace(final_path)
        except MLXUserError:
            raise
        except (
            AttributeError,
            FileNotFoundError,
            ImportError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise MLXUserError(
                f"LibreYOLO ONNX export failed: {exc}. Check the checkpoint and install the "
                "object-detection-libreyolo extra with ONNX export dependencies."
            ) from exc

        emit(self.reporter, "success", f"LibreYOLO ONNX export complete: {final_path}")
        return final_path


def convert_object_detection_model(config: dict[str, Any]) -> Path:
    """Compatibility function for direct LibreYOLO-provider callers."""

    from mlx.modes.object_detection.presentation import RichWorkflowReporter

    return ConvertLibreYOLOObjectDetectionModel(
        config,
        reporter=RichWorkflowReporter(),
    ).execute()
