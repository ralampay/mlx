from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.modes.object_detection.models import DetectionAdapter, DetectionResult
from mlx.modes.object_detection.providers import ObjectDetectionProvider, get_provider
from mlx.modes.object_detection.requests import (
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
        provider = self.provider or get_provider(self.request.provider)
        return provider.train(self.request, self.reporter)


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

