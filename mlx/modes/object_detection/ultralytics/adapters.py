from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from mlx.modes.object_detection.ultralytics.utils import initialize_model

try:
    import onnxruntime  # noqa: F401
except ImportError:
    onnxruntime = None

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "The ultralytics package (ralampay fork) is required for object-detection mode."
    ) from exc


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    names: dict[int, str]


class DetectionAdapter(Protocol):
    def predict(self, frame: np.ndarray) -> DetectionResult:
        ...


class UltralyticsDetectionAdapter:
    def __init__(
        self,
        *,
        resolved_cfg: Optional[Path],
        resolved_weights: Path,
        device: str,
        imgsz: int | tuple[int, int],
        confidence: float,
    ) -> None:
        self.model = initialize_model(resolved_cfg, resolved_weights, prefer_cfg=False)
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence

    def predict(self, frame: np.ndarray) -> DetectionResult:
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            stream=False,
        )[0]
        return _result_to_detection_result(result)


class OnnxRuntimeDetectionAdapter:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        imgsz: int | tuple[int, int],
        confidence: float,
    ) -> None:
        if onnxruntime is None:
            raise ImportError(
                "onnxruntime is required for ONNX object-detection inference. Install it with 'pip install onnxruntime'."
            )

        # Let Ultralytics drive ONNX Runtime inference so preprocessing, decoding,
        # and NMS stay identical to the reference `.pt` path.
        self.model = YOLO(str(model_path), task="detect")
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence

    def predict(self, frame: np.ndarray) -> DetectionResult:
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            stream=False,
        )[0]
        return _result_to_detection_result(result)


def build_detection_adapter(
    *,
    resolved_cfg: Optional[Path],
    resolved_weights: Path,
    device: str,
    imgsz: int | tuple[int, int],
    confidence: float,
) -> DetectionAdapter:
    suffix = resolved_weights.suffix.lower()
    if suffix == ".onnx":
        return OnnxRuntimeDetectionAdapter(
            model_path=resolved_weights,
            device=device,
            imgsz=imgsz,
            confidence=confidence,
        )
    return UltralyticsDetectionAdapter(
        resolved_cfg=resolved_cfg,
        resolved_weights=resolved_weights,
        device=device,
        imgsz=imgsz,
        confidence=confidence,
    )


def _result_to_detection_result(result) -> DetectionResult:
    names = {int(key): str(value) for key, value in (result.names or {}).items()}
    detections: list[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return DetectionResult(detections=detections, names=names)

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), score, class_id in zip(xyxy, confs, classes):
        detections.append(
            Detection(
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(score),
                class_id=int(class_id),
                label=names.get(int(class_id), str(int(class_id))),
            )
        )
    return DetectionResult(detections=detections, names=names)
