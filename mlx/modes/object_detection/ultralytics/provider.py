from __future__ import annotations

from mlx.core.commands import WorkflowReporter, emit
from mlx.modes.object_detection.requests import (
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    TrainObjectDetectionRequest,
)


class UltralyticsProvider:
    name = "ultralytics"

    def train(self, request: TrainObjectDetectionRequest, reporter: WorkflowReporter):
        from mlx.modes.object_detection.ultralytics.training import (
            TrainUltralyticsObjectDetection,
        )

        emit(reporter, "info", "Starting Ultralytics object-detection training.")

        def report_checkpoint(path) -> None:
            emit(
                reporter,
                "progress",
                "Ultralytics completed a recoverable training epoch.",
                payload={"checkpoint_path": str(path)},
            )

        return TrainUltralyticsObjectDetection(
            request.to_config(),
            checkpoint_observer=report_checkpoint,
        ).execute()

    def create_detector(self, request: ObjectDetectionRequest):
        from mlx.modes.object_detection.ultralytics.adapters import build_detection_adapter
        from mlx.modes.object_detection.ultralytics.utils import resolve_imgsz, resolve_model_paths

        config = request.to_config()
        resolved_cfg, resolved_weights = resolve_model_paths(
            config,
            require_yaml=True,
            require_weights=True,
        )
        return build_detection_adapter(
            resolved_cfg=resolved_cfg,
            resolved_weights=resolved_weights,
            device=request.device,
            imgsz=resolve_imgsz(config),
            confidence=request.confidence,
        )

    def convert(
        self,
        request: ConvertObjectDetectionRequest,
        reporter: WorkflowReporter,
    ):
        from mlx.modes.object_detection.ultralytics.conversion import (
            ConvertUltralyticsObjectDetectionModel,
        )

        emit(reporter, "info", "Exporting an Ultralytics checkpoint.")
        return ConvertUltralyticsObjectDetectionModel(request.to_config()).execute()

    def list_models(
        self,
        request: ListObjectDetectionModelsRequest,
        reporter: WorkflowReporter,
    ):
        from mlx.modes.object_detection.ultralytics.list_models import (
            ListObjectDetectionModels,
        )

        return tuple(ListObjectDetectionModels().execute())


def get_provider() -> UltralyticsProvider:
    return UltralyticsProvider()
