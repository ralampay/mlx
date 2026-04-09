from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_warning
from mlx.modes.segmentation.data import load_image_tensor
from mlx.modes.segmentation.presentation import (
    blend_overlay,
    colorize_mask,
    display_segmentation_result,
    stack_segmentation_views,
)
from mlx.modes.segmentation.utils import load_checkpoint_bundle


def infer_segmentation_image(config: dict[str, Any]) -> dict[str, Any]:
    model, metadata = load_checkpoint_bundle(config)
    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    input_img_path = Path(config["input_img"])
    if not input_img_path.exists():
        raise MLXUserError(f"Input image not found: {input_img_path}")

    tensor = load_image_tensor(
        input_img_path,
        input_size=metadata["input_size"],
        colored=metadata["colored"],
    ).unsqueeze(0).to(device)

    original_bgr = cv2.imread(str(input_img_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise MLXUserError(f"Unable to read input image: {input_img_path}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        logits = model(tensor)
        predicted_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_mask = colorize_mask(predicted_mask, metadata["palette"])
    overlay_rgb = blend_overlay(
        cv2.resize(original_rgb, metadata["input_size"], interpolation=cv2.INTER_LINEAR),
        color_mask,
        float(config.get("overlay_alpha", metadata.get("overlay_alpha", 0.45))),
    )
    window_image = stack_segmentation_views(
        cv2.resize(original_rgb, metadata["input_size"], interpolation=cv2.INTER_LINEAR),
        predicted_mask,
        overlay_rgb,
        palette=metadata["palette"],
    )

    result = {
        "input_image": input_img_path,
        "input_size": metadata["input_size"],
        "model_name": metadata["model_name"],
        "num_classes": metadata["num_classes"],
        "predicted_mask": predicted_mask,
        "window_image": window_image,
    }
    display_segmentation_result(result)
    return result


class StreamSegmentationInferenceRunner:
    def __init__(self, config: dict[str, Any], source: str) -> None:
        self.config = config
        self.source = source
        self.device = config.get("device", "cpu")
        self.camera_index = int(config.get("camera_index", 0))
        self.overlay_alpha = float(config.get("overlay_alpha", 0.45))
        self.model, self.metadata = load_checkpoint_bundle(config)
        self.model = self.model.to(self.device)
        self.model.eval()

    def execute(self) -> None:
        print_info(
            f"Using device: {self.device} | Input size: {self.metadata['input_size'][0]}x{self.metadata['input_size'][1]}"
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
                rendered = self._render_frame(frame)
                cv2.imshow(window_title, rendered)
                key = cv2.waitKey(1 if self.source == "camera" else 10) & 0xFF
                if key in (ord("q"), 27):
                    print_info("Exiting inference.")
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _render_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(
                cv2.resize(frame_rgb, self.metadata["input_size"], interpolation=cv2.INTER_LINEAR)
                .transpose(2, 0, 1)
            )
            .float()
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )
        with torch.no_grad():
            logits = self.model(tensor)
            predicted_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        resized_rgb = cv2.resize(frame_rgb, self.metadata["input_size"], interpolation=cv2.INTER_LINEAR)
        color_mask = colorize_mask(predicted_mask, self.metadata["palette"])
        overlay_rgb = blend_overlay(resized_rgb, color_mask, self.overlay_alpha)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        if self.metadata["num_classes"] <= 2:
            foreground = (predicted_mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)

        return overlay_bgr

    def _open_capture(self):
        if self.source == "camera":
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open camera index {self.camera_index}.")
            return capture, "MLX Segmentation (Camera)"

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
            return capture, f"MLX Segmentation (Video: {resolved_video.name})"

        raise MLXUserError(f"Unsupported source type: {self.source}")

