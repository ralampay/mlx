from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.models import (
    DetectionAdapter,
    DetectionResult,
    ObjectDetectionTrainingResult,
)
from mlx.modes.object_detection.providers import ObjectDetectionProvider, get_provider
from mlx.modes.object_detection.requests import (
    BenchmarkObjectDetectionRequest,
    ConvertObjectDetectionRequest,
    ListObjectDetectionModelsRequest,
    ObjectDetectionRequest,
    TrainObjectDetectionRequest,
)
from mlx.modes.object_detection.streaming import FrameSink, FrameSource, StreamInferenceResult


class TrainObjectDetectionModel:
    def __init__(
        self,
        request: TrainObjectDetectionRequest,
        *,
        provider: Optional[ObjectDetectionProvider] = None,
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.request = request
        self.provider = provider
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> Any:
        if self.request.validate_after_training:
            _validate_post_training_benchmark(self.request)
        provider = self.provider or get_provider(self.request.provider)
        training_result = provider.train(self.request, self.reporter)
        if not self.request.validate_after_training:
            return training_result

        checkpoint_path = _training_checkpoint_path(training_result)
        run_dir = (
            checkpoint_path.parent.parent
            if checkpoint_path.parent.name == "weights"
            else checkpoint_path.parent
        )
        benchmark_output = run_dir / "benchmark" / self.request.validation_split
        benchmark_request = BenchmarkObjectDetectionRequest(
            provider=self.request.provider,
            model=self.request.model,
            model_path=str(checkpoint_path),
            device=self.request.device,
            height=self.request.height,
            width=self.request.width,
            confidence=self.request.validation_confidence,
            dataset_path=self.request.dataset_path,
            output_path=str(benchmark_output),
            batch_size=self.request.batch_size,
            split=self.request.validation_split,
            iou=self.request.validation_iou,
            max_detections=self.request.validation_max_detections,
            plots=self.request.plots,
        )
        benchmark_result = BenchmarkObjectDetectionModel(
            benchmark_request,
            provider=provider,
            reporter=self.reporter,
        ).execute()
        return ObjectDetectionTrainingResult(
            training_result=training_result,
            checkpoint_path=str(checkpoint_path),
            benchmark_result=benchmark_result,
        )


class BenchmarkObjectDetectionModel:
    def __init__(
        self,
        request: BenchmarkObjectDetectionRequest,
        *,
        provider: Optional[ObjectDetectionProvider] = None,
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.request = request
        self.provider = provider
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self):
        provider = self.provider or get_provider(self.request.provider)
        return provider.benchmark(self.request, self.reporter)


class CreateObjectDetector:
    def __init__(
        self,
        request: ObjectDetectionRequest,
        *,
        provider: Optional[ObjectDetectionProvider] = None,
    ) -> None:
        self.request = request
        self.provider = provider

    def execute(self) -> DetectionAdapter:
        provider = self.provider or get_provider(self.request.provider)
        return provider.create_detector(self.request)


class ConvertObjectDetectionModel:
    def __init__(
        self,
        request: ConvertObjectDetectionRequest,
        *,
        provider: Optional[ObjectDetectionProvider] = None,
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.request = request
        self.provider = provider
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> Path:
        provider = self.provider or get_provider(self.request.provider)
        return provider.convert(self.request, self.reporter)


class ListObjectDetectionModels:
    def __init__(
        self,
        request: ListObjectDetectionModelsRequest,
        *,
        provider: Optional[ObjectDetectionProvider] = None,
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.request = request
        self.provider = provider
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self):
        provider = self.provider or get_provider(self.request.provider)
        return tuple(provider.list_models(self.request, self.reporter))


class RunObjectDetectionStream:
    def __init__(
        self,
        *,
        detector: DetectionAdapter,
        frame_source: FrameSource,
        frame_sink: FrameSink,
        renderer: Callable[[np.ndarray, DetectionResult], np.ndarray],
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.detector = detector
        self.frame_source = frame_source
        self.frame_sink = frame_sink
        self.renderer = renderer
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> StreamInferenceResult:
        frames_processed = 0
        stopped_by_user = False
        try:
            while True:
                ok, frame = self.frame_source.read()
                if not ok:
                    break
                result = self.detector.predict(frame)
                frames_processed += 1
                if not self.frame_sink.show(self.renderer(frame, result)):
                    stopped_by_user = True
                    break
        finally:
            self.frame_source.release()
            self.frame_sink.close()

        emit(
            self.reporter,
            "success",
            f"Object-detection stream processed {frames_processed} frame(s).",
        )
        return StreamInferenceResult(
            frames_processed=frames_processed,
            stopped_by_user=stopped_by_user,
        )


def _training_checkpoint_path(result: Any) -> Path:
    if isinstance(result, dict):
        candidate = result.get("checkpoint_path") or result.get("model_path")
    else:
        candidate = getattr(result, "checkpoint_path", None) or getattr(
            result, "model_path", None
        )
    if not candidate:
        raise MLXUserError(
            "Training completed without a selected checkpoint, so post-training "
            "benchmarking could not run."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise MLXUserError(
            f"Training selected checkpoint is unavailable for benchmarking: {path}"
        )
    return path


def _validate_post_training_benchmark(request: TrainObjectDetectionRequest) -> None:
    if request.validation_split not in {"train", "val", "test"}:
        raise MLXUserError("Post-training validation split must be train, val, or test.")
    if not 0.0 <= request.validation_confidence <= 1.0:
        raise MLXUserError("Post-training validation confidence must be between 0 and 1.")
    if not 0.0 < request.validation_iou <= 1.0:
        raise MLXUserError(
            "Post-training validation IoU must be greater than 0 and at most 1."
        )
    if request.validation_max_detections < 1:
        raise MLXUserError(
            "Post-training validation max detections must be at least 1."
        )
