from __future__ import annotations

from mlx.core.commands import WorkflowReporter, emit
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    TrainObjectDetectionRequest,
)


class LibreYOLOProvider:
    name = "libreyolo"

    def train(self, request: TrainObjectDetectionRequest, reporter: WorkflowReporter):
        from mlx.modes.object_detection.libreyolo.training import TrainLibreYOLOObjectDetection

        emit(reporter, "info", "Starting LibreYOLO object-detection training.")
        return TrainLibreYOLOObjectDetection(request, reporter=reporter).execute()

    def benchmark(
        self,
        request: BenchmarkObjectDetectionRequest,
        reporter: WorkflowReporter,
    ):
        from mlx.modes.object_detection.libreyolo.evaluation import (
            BenchmarkLibreYOLOObjectDetection,
        )

        emit(reporter, "info", "Benchmarking a LibreYOLO object detector.")
        result = BenchmarkLibreYOLOObjectDetection(request).execute()
        emit(
            reporter,
            "success",
            f"Object-detection benchmark artifacts written to {result.output_dir}.",
        )
        return result

    def create_detector(self, request: ObjectDetectionRequest):
        from mlx.modes.object_detection.libreyolo.adapters import build_detection_adapter
        from mlx.modes.object_detection.libreyolo.utils import resolve_imgsz, resolve_model_path

        model_path = resolve_model_path(request.model_path, required=True)
        return build_detection_adapter(
            model_path=model_path,
            device=request.device,
            imgsz=resolve_imgsz(request.to_config()),
            confidence=request.confidence,
        )

    def convert(
        self,
        request: ConvertObjectDetectionRequest,
        reporter: WorkflowReporter,
    ):
        from mlx.modes.object_detection.libreyolo.conversion import (
            ConvertLibreYOLOObjectDetectionModel,
        )

        emit(reporter, "info", "Exporting a LibreYOLO checkpoint.")
        return ConvertLibreYOLOObjectDetectionModel(request, reporter=reporter).execute()

    def list_models(
        self,
        request: ListObjectDetectionModelsRequest,
        reporter: WorkflowReporter,
    ):
        from mlx.modes.object_detection.libreyolo.list_models import ListLibreYOLOModels

        return tuple(ListLibreYOLOModels().execute())


def get_provider() -> LibreYOLOProvider:
    return LibreYOLOProvider()
