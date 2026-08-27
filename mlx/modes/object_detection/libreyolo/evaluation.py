from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import time

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.evaluation import (
    resolve_benchmark_output_dir,
    validate_benchmark_request,
    write_benchmark_artifacts,
)
from mlx.modes.object_detection.libreyolo.utils import (
    dependency_error,
    resolve_dataset_source,
    resolve_imgsz,
    resolve_model_path,
)
from mlx.modes.object_detection.requests import BenchmarkObjectDetectionRequest


class BenchmarkLibreYOLOObjectDetection:
    def __init__(self, request: BenchmarkObjectDetectionRequest) -> None:
        self.request = request

    def execute(self):
        validate_benchmark_request(self.request)
        try:
            from libreyolo import LibreYOLO
        except ImportError as exc:
            raise dependency_error("benchmarking an object-detection model") from exc

        config = self.request.to_config()
        model_path = resolve_model_path(self.request.model_path, required=True)
        dataset = resolve_dataset_source({**config, "output_path": None})
        output_dir = resolve_benchmark_output_dir(self.request)
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        try:
            model = LibreYOLO(str(model_path), device=self.request.device, task="detect")
            metrics = model.val(
                batch=self.request.batch_size,
                conf=self.request.confidence,
                data=dataset.data,
                device=self.request.device,
                eval_max_det=self.request.max_detections,
                imgsz=resolve_imgsz(config),
                iou=self.request.iou,
                max_det=self.request.max_detections,
                save_dir=str(output_dir),
                save_json=self.request.save_predictions,
                save_plots=self.request.plots,
                split=self.request.split,
                verbose=False,
                workers=self.request.workers,
            )
        except Exception as exc:
            raise MLXUserError(
                "LibreYOLO object-detection benchmark failed: "
                f"{exc}. Check the model, dataset split, device, and evaluation options."
            ) from exc

        backend = str(getattr(model, "last_eval_backend", None) or _version("libreyolo"))
        return write_benchmark_artifacts(
            request=self.request,
            provider="libreyolo",
            dataset=dataset.source,
            evaluation_backend=backend,
            raw_metrics=metrics,
            started_at=started_at,
            started_clock=started_clock,
            provider_version=_version("libreyolo"),
        )


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"
