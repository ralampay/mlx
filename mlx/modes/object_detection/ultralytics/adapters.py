from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.ultralytics.utils import initialize_model

try:
    import onnxruntime as ort
except ImportError:
    ort = None


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

        names = {int(key): str(value) for key, value in (result.names or {}).items()}
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(detections=detections, names=names)

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros(len(xyxy))
        classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes.cls is not None
            else np.zeros(len(xyxy), dtype=int)
        )

        for (x1, y1, x2, y2), score, class_id in zip(xyxy, confs, classes):
            label = names.get(int(class_id), str(int(class_id)))
            detections.append(
                Detection(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(score),
                    class_id=int(class_id),
                    label=label,
                )
            )

        return DetectionResult(detections=detections, names=names)


class OnnxRuntimeDetectionAdapter:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        imgsz: int | tuple[int, int],
        confidence: float,
        iou_threshold: float = 0.45,
    ) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX object-detection inference. Install it with 'pip install onnxruntime'."
            )

        self.model_path = model_path
        self.device = device
        self.input_height, self.input_width = _normalize_imgsz(imgsz)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(str(model_path), providers=_resolve_providers(device))
        self.input_name = self.session.get_inputs()[0].name
        self.names = _parse_names_metadata(self.session.get_modelmeta().custom_metadata_map)

    def predict(self, frame: np.ndarray) -> DetectionResult:
        input_tensor, scale, pad_x, pad_y = _preprocess_frame(
            frame,
            target_height=self.input_height,
            target_width=self.input_width,
        )
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self._postprocess(outputs[0], frame.shape[:2], scale, pad_x, pad_y)
        return DetectionResult(detections=detections, names=self.names)

    def _postprocess(
        self,
        raw_output: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        predictions = np.asarray(raw_output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise MLXUserError(
                f"Unsupported ONNX output shape {tuple(np.asarray(raw_output).shape)} for detection inference."
            )
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()

        if predictions.shape[1] < 6:
            raise MLXUserError(
                f"Unsupported ONNX prediction tensor shape {tuple(predictions.shape)} for detection inference."
            )

        # Handle a graph that already emitted final detections: [x1, y1, x2, y2, score, class_id].
        if _looks_like_final_detections(predictions):
            return _build_final_detections(
                predictions,
                original_shape=original_shape,
                names=self.names,
                confidence=self.confidence,
            )

        boxes_xyxy, scores, class_ids = _decode_raw_predictions(
            predictions,
            names=self.names,
            confidence=self.confidence,
        )
        if len(boxes_xyxy) == 0:
            return []

        boxes_xyxy = _scale_boxes_to_original(
            boxes_xyxy,
            original_shape=original_shape,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        keep = _non_max_suppression(boxes_xyxy, scores, class_ids, self.iou_threshold)

        detections: list[Detection] = []
        for index in keep:
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    xyxy=tuple(int(v) for v in boxes_xyxy[index]),
                    confidence=float(scores[index]),
                    class_id=class_id,
                    label=self.names.get(class_id, str(class_id)),
                )
            )
        return detections


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


def _normalize_imgsz(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return imgsz, imgsz
    return int(imgsz[0]), int(imgsz[1])


def _resolve_providers(device: str) -> list[str]:
    device_name = str(device).lower()
    available = ort.get_available_providers() if ort is not None else []
    if device_name.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise MLXUserError(
                "CUDA was requested for ONNX inference, but onnxruntime does not expose CUDAExecutionProvider in this environment."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _parse_names_metadata(metadata: dict[str, str]) -> dict[int, str]:
    raw_names = metadata.get("names") if metadata else None
    if not raw_names:
        return {}

    try:
        parsed = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError):
        return {}

    if isinstance(parsed, dict):
        return {int(key): str(value) for key, value in parsed.items()}
    if isinstance(parsed, list):
        return {index: str(value) for index, value in enumerate(parsed)}
    return {}


def _preprocess_frame(
    frame: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, float, float, float]:
    original_height, original_width = frame.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized = frame
    if (resized_width, resized_height) != (original_width, original_height):
        import cv2

        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_w = target_width - resized_width
    pad_h = target_height - resized_height
    pad_left = pad_w / 2.0
    pad_top = pad_h / 2.0
    left = int(np.floor(pad_left))
    top = int(np.floor(pad_top))

    padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    padded[top : top + resized_height, left : left + resized_width] = resized
    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0), scale, float(left), float(top)


def _looks_like_final_detections(predictions: np.ndarray) -> bool:
    if predictions.shape[1] not in {6, 7}:
        return False
    class_column = predictions[:, 5]
    if not np.all(np.isfinite(class_column)):
        return False
    rounded = np.rint(class_column)
    return np.all(np.abs(class_column - rounded) < 1e-3)


def _build_final_detections(
    predictions: np.ndarray,
    *,
    original_shape: tuple[int, int],
    names: dict[int, str],
    confidence: float,
) -> list[Detection]:
    height, width = original_shape
    detections: list[Detection] = []
    for row in predictions:
        score = float(row[4])
        if score < confidence:
            continue
        class_id = int(row[5])
        x1, y1, x2, y2 = row[:4]
        detections.append(
            Detection(
                xyxy=(
                    int(np.clip(x1, 0, width - 1)),
                    int(np.clip(y1, 0, height - 1)),
                    int(np.clip(x2, 0, width - 1)),
                    int(np.clip(y2, 0, height - 1)),
                ),
                confidence=score,
                class_id=class_id,
                label=names.get(class_id, str(class_id)),
            )
        )
    return detections


def _decode_raw_predictions(
    predictions: np.ndarray,
    *,
    names: dict[int, str],
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if names and predictions.shape[1] == 4 + len(names):
        class_scores = predictions[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
    elif names and predictions.shape[1] == 5 + len(names):
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = objectness * class_scores[np.arange(len(predictions)), class_ids]
    else:
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(predictions)), class_ids]

    keep = scores >= confidence
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    boxes_xywh = predictions[keep, :4]
    scores = scores[keep]
    class_ids = class_ids[keep].astype(np.int32)
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return boxes_xyxy, scores, class_ids


def _scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    *,
    original_shape: tuple[int, int],
    scale: float,
    pad_x: float,
    pad_y: float,
) -> np.ndarray:
    height, width = original_shape
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def _non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    keep: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while len(order) > 0:
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            ious = _compute_iou(boxes_xyxy[current], boxes_xyxy[remaining])
            order = remaining[ious <= iou_threshold]
    keep.sort(key=lambda index: scores[index], reverse=True)
    return keep


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
