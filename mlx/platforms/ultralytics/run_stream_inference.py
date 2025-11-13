try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for --action infer-camera. Install it with 'pip install opencv-python'."
    ) from exc

import typer
from typing import Dict, Any, Optional, Tuple, Union
from mlx.platforms.ultralytics.utils import _resolve_model_paths, _initialize_model, _annotate_detections

class RunStreamInference:
    def __init__(self, config: Dict[str, Any], source: str):
        self.config = config
        self.source = source

        self.device = config.get("device", "cpu")

        self.imgsz = max(
            self.config.get("height", 640), 
            config.get("width", 640)
        )

        self.resolved_cfg, self.resolved_weights = _resolve_model_paths(
            config, require_yaml=True, require_weights=True
        )

        self.confidence = float(self.config.get("confidence", 0.25))

        self.camera_index = int(self.config.get("camera_index", 0))

        if source == "camera":
            typer.secho(
                "Ultralytics Object Detection - Camera Inference", 
                fg=typer.colors.BRIGHT_CYAN, bold=True
            )
        else:
            typer.secho(
                "Ultralytics Object Detection - Video Inference", 
                fg=typer.colors.BRIGHT_CYAN, bold=True
            )

        if self.resolved_cfg:
            typer.echo(f"Model YAML: {self.resolved_cfg}")
        typer.echo(f"Loading weights from: {self.resolved_weights}")

        self.model = _initialize_model(
            self.resolved_cfg, 
            self.resolved_weights, 
            prefer_cfg=False
        )

    def execute(self):
        typer.echo(
            f"Using device: {self.device} | Image size: {self.imgsz} | Confidence: {self.confidence}"
        )
        typer.secho("Press 'q' or 'Esc' to exit.", fg=typer.colors.YELLOW)

        if self.source == "camera":
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open camera index {self.camera_index}.")
            window_title = "MLX Object Detection (Camera)"
        elif self.source == "video":
            video_path = self.config.get("file_path")
            if not video_path:
                raise typer.BadParameter("Video inference requires --file-path pointing to the video file.")
            resolved_video = Path(video_path).expanduser()

            if not resolved_video.exists():
                raise typer.BadParameter(f"Video file not found: {resolved_video}")

            cap = cv2.VideoCapture(str(resolved_video))

            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video file: {resolved_video}")

            window_title = f"MLX Object Detection (Video: {resolved_video.name})"
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    typer.echo("No more frames to process." if self.source == "video" else "Failed to read frame from camera.")
                    break

                result = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.confidence,
                    device=self.device,
                    verbose=False,
                    stream=False,
                )

                annotated = _annotate_detections(frame, result[0])
                cv2.imshow(window_title, annotated)

                key = cv2.waitKey(1 if self.source == "camera" else 10) & 0xFF
                if key in (ord("q"), 27):
                    typer.echo("Exiting inference.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
