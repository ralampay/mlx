from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.panel import Panel

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_warning
from mlx.modes.object_detection.ultralytics.adapters import build_detection_adapter
from mlx.modes.object_detection.ultralytics.utils import (
    annotate_detections,
    resolve_imgsz,
    resolve_model_paths,
)

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for object-detection inference. Install it with 'pip install opencv-python'."
    ) from exc


class StreamInferenceRunner:
    def __init__(self, config: dict[str, Any], source: str) -> None:
        self.config = config
        self.source = source
        self.device = config.get("device", "cpu")
        self.imgsz = resolve_imgsz(config)
        self.resolved_cfg, self.resolved_weights = resolve_model_paths(
            config,
            require_yaml=True,
            require_weights=True,
        )
        self.confidence = float(config.get("confidence", 0.25))
        self.camera_index = int(config.get("camera_index", 0))

        title = (
            "Ultralytics Object Detection - Camera Inference"
            if source == "camera"
            else "Ultralytics Object Detection - Video Inference"
        )
        console.print(Panel.fit(title, border_style="cyan"))

        if self.resolved_cfg:
            print_info(f"Model YAML: {self.resolved_cfg}")
        print_info(f"Loading weights from: {self.resolved_weights}")
        self.adapter = build_detection_adapter(
            resolved_cfg=self.resolved_cfg,
            resolved_weights=self.resolved_weights,
            device=self.device,
            imgsz=self.imgsz,
            confidence=self.confidence,
        )

    def execute(self) -> None:
        print_info(
            f"Using device: {self.device} | Image size: {self.imgsz} | Confidence: {self.confidence}"
        )
        print_warning("Press 'q' or 'Esc' to exit.")
        capture, window_title = self._open_capture()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print_warning(
                        "No more frames to process."
                        if self.source == "video"
                        else "Failed to read frame from camera."
                    )
                    break

                result = self.adapter.predict(frame)
                cv2.imshow(window_title, annotate_detections(frame, result))
                key = cv2.waitKey(1 if self.source == "camera" else 10) & 0xFF
                if key in (ord("q"), 27):
                    print_info("Exiting inference.")
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _open_capture(self):
        if self.source == "camera":
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open camera index {self.camera_index}.")
            return capture, "MLX Object Detection (Camera)"

        if self.source == "video":
            video_path = self.config.get("file_path")
            if not video_path:
                raise MLXUserError("Video inference requires --file-path pointing to the video file.")
            resolved_video = Path(video_path).expanduser()
            if not resolved_video.exists():
                raise MLXUserError(f"Video file not found: {resolved_video}")

            capture = cv2.VideoCapture(str(resolved_video))
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open video file: {resolved_video}")
            return capture, f"MLX Object Detection (Video: {resolved_video.name})"

        raise MLXUserError(f"Unsupported source type: {self.source}")
