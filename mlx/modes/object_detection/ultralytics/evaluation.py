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
from mlx.modes.object_detection.requests import BenchmarkObjectDetectionRequest
from mlx.modes.object_detection.ultralytics.utils import (
    initialize_model,
    resolve_dataset_source,
    resolve_imgsz,
    resolve_model_paths,
)


class BenchmarkUltralyticsObjectDetection:
    def __init__(self, request: BenchmarkObjectDetectionRequest) -> None:
        self.request = request

    def execute(self):
        validate_benchmark_request(self.request)
        config = self.request.to_config()
        _, weights = resolve_model_paths(
            config,
            require_yaml=False,
            require_weights=True,
        )
        dataset = resolve_dataset_source({**config, "output_path": None})
        output_dir = resolve_benchmark_output_dir(self.request)
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        try:
            model = initialize_model(None, weights, prefer_cfg=False)
            results = model.val(
                batch=self.request.batch_size,
                conf=self.request.confidence,
                data=dataset.data,
                device=self.request.device,
                exist_ok=True,
                imgsz=resolve_imgsz(config),
                iou=self.request.iou,
                max_det=self.request.max_detections,
                name=output_dir.name,
                plots=self.request.plots,
                project=str(output_dir.parent),
                save_json=self.request.save_predictions,
                split=self.request.split,
                verbose=False,
                workers=self.request.workers,
            )
        except Exception as exc:
            raise MLXUserError(
                "Ultralytics object-detection benchmark failed: "
                f"{exc}. Check the model, dataset split, device, and evaluation options."
            ) from exc

        provider_version = _version("ultralytics")
        return write_benchmark_artifacts(
            request=self.request,
            provider="ultralytics",
            dataset=dataset.source,
            evaluation_backend=f"ultralytics {provider_version}",
            raw_metrics=results,
            started_at=started_at,
            started_clock=started_clock,
            provider_version=provider_version,
        )


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"
