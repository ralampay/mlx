from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.models import Detection, DetectionAdapter, DetectionResult
from mlx.modes.object_detection.libreyolo.utils import dependency_error


class LibreYOLODetectionAdapter:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        imgsz: int | tuple[int, int],
        confidence: float,
    ) -> None:
        try:
            from libreyolo import LibreYOLO
        except ImportError as exc:
            raise dependency_error("running object-detection inference") from exc

        try:
            self.model = LibreYOLO(str(model_path), device=device, task="detect")
        except (
            AttributeError,
            FileNotFoundError,
            ImportError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise MLXUserError(
                f"Failed to load LibreYOLO detection model '{model_path}': {exc}. "
                "Verify that the checkpoint is a supported axis-aligned detection model."
            ) from exc
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence

    def predict(self, frame: np.ndarray) -> DetectionResult:
        try:
            raw_result = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                device=self.device,
                stream=False,
                color_format="bgr",
            )
            result = _unwrap_single_result(raw_result)
            return result_to_detection_result(result)
        except MLXUserError:
            raise
        except (AttributeError, ImportError, TypeError, ValueError, RuntimeError) as exc:
            raise MLXUserError(
                f"LibreYOLO inference failed: {exc}. Check the checkpoint, device, and image size."
            ) from exc


def build_detection_adapter(
    *,
    model_path: Path,
    device: str,
    imgsz: int | tuple[int, int],
    confidence: float,
) -> DetectionAdapter:
    if model_path.suffix.lower() not in {".pt", ".onnx"}:
        raise MLXUserError(
            "LibreYOLO inference currently supports --model-path files ending in .pt or .onnx."
        )
    return LibreYOLODetectionAdapter(
        model_path=model_path,
        device=device,
        imgsz=imgsz,
        confidence=confidence,
    )


def _unwrap_single_result(raw_result: Any) -> Any:
    if isinstance(raw_result, (list, tuple)):
        if not raw_result:
            raise MLXUserError("LibreYOLO inference returned no result for the input frame.")
        return raw_result[0]
    if isinstance(raw_result, Iterator):
        try:
            return next(raw_result)
        except StopIteration as exc:
            raise MLXUserError(
                "LibreYOLO inference returned no result for the input frame."
            ) from exc
    return raw_result


def result_to_detection_result(result: Any) -> DetectionResult:
    if result is None:
        raise MLXUserError("LibreYOLO inference returned an invalid empty result.")

    names = _normalize_names(getattr(result, "names", None))
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return DetectionResult(detections=(), names=names)

    xyxy = _to_numpy(getattr(boxes, "xyxy", None), field_name="boxes.xyxy")
    if xyxy.size == 0:
        return DetectionResult(detections=(), names=names)
    xyxy = xyxy.reshape(-1, 4)

    confs = _to_numpy(getattr(boxes, "conf", None), field_name="boxes.conf").reshape(-1)
    classes = _to_numpy(getattr(boxes, "cls", None), field_name="boxes.cls").reshape(-1)
    if len(xyxy) != len(confs) or len(xyxy) != len(classes):
        raise MLXUserError(
            "LibreYOLO returned inconsistent detection arrays for boxes, confidence, and classes."
        )

    detections = []
    for coordinates, score, raw_class_id in zip(xyxy, confs, classes):
        class_id = int(raw_class_id)
        detections.append(
            Detection(
                xyxy=tuple(float(value) for value in coordinates),
                confidence=float(score),
                class_id=class_id,
                label=names.get(class_id, str(class_id)),
            )
        )
    return DetectionResult(detections=tuple(detections), names=names)


def _normalize_names(raw_names: Any) -> dict[int, str]:
    if raw_names is None:
        return {}
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise MLXUserError("LibreYOLO returned class names in an unsupported format.")


def _to_numpy(value: Any, *, field_name: str) -> np.ndarray:
    if value is None:
        raise MLXUserError(f"LibreYOLO result is missing required {field_name} values.")
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    array_value = cpu_value.numpy() if hasattr(cpu_value, "numpy") else cpu_value
    return np.asarray(array_value)
