from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.data import load_image_tensor
from mlx.modes.segmentation.streaming import (
    SegmentationFrameSink,
    SegmentationFrameSource,
    SegmentationStreamResult,
)
from mlx.modes.segmentation.visualization import (
    blend_overlay,
    colorize_mask,
    stack_segmentation_views,
)
from mlx.modes.segmentation.utils import load_checkpoint_bundle
from mlx.modes.segmentation.requests import SegmentationRequest


class InferSegmentationImage:
    def __init__(self, request: SegmentationRequest) -> None:
        self.request = request

    def execute(self) -> dict[str, Any]:
        return _run_image_inference(self.request.to_config())


def infer_segmentation_image(config: dict[str, Any]) -> dict[str, Any]:
    compatibility_config = {"display": True, **config}
    result = InferSegmentationImage(
        SegmentationRequest.from_config(compatibility_config)
    ).execute()
    if compatibility_config.get("display", True):
        from mlx.modes.segmentation.presentation import display_segmentation_result

        display_segmentation_result(result)
    return result


def _run_image_inference(config: dict[str, Any]) -> dict[str, Any]:
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
    return result


class RunSegmentationStreamInference:
    def __init__(
        self,
        config: dict[str, Any] | SegmentationRequest,
        source: str,
        *,
        frame_source: SegmentationFrameSource | None = None,
        frame_sink: SegmentationFrameSink | None = None,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        if isinstance(config, SegmentationRequest):
            config = config.to_config()
        self.config = config
        self.source = source
        self.device = config.get("device", "cpu")
        self.camera_index = int(config.get("camera_index", 0))
        self.overlay_alpha = float(config.get("overlay_alpha", 0.45))
        self.frame_source = frame_source
        self.frame_sink = frame_sink
        self.reporter = reporter or NullWorkflowReporter()
        self.model = None
        self.metadata: dict[str, Any] = {}

    def execute(self) -> SegmentationStreamResult:
        if self.frame_source is None or self.frame_sink is None:
            raise MLXUserError(
                "Segmentation stream inference requires injected frame source and sink adapters."
            )
        self.model, self.metadata = load_checkpoint_bundle(self.config)
        self.model = self.model.to(self.device)
        self.model.eval()
        emit(
            self.reporter,
            "info",
            f"Using device: {self.device} | Input size: {self.metadata['input_size'][0]}x{self.metadata['input_size'][1]}"
        )
        frames_processed = 0
        stopped_by_user = False
        try:
            while True:
                ok, frame = self.frame_source.read()
                if not ok or frame is None:
                    emit(
                        self.reporter,
                        "warning",
                        "No more frames to process."
                        if self.source == "video"
                        else "Failed to read frame from camera.",
                    )
                    break
                rendered = self._render_frame(frame)
                frames_processed += 1
                if not self.frame_sink.show(rendered):
                    stopped_by_user = True
                    emit(self.reporter, "info", "Exiting inference.")
                    break
        finally:
            self.frame_source.release()
            self.frame_sink.close()
        return SegmentationStreamResult(
            frames_processed=frames_processed,
            stopped_by_user=stopped_by_user,
        )

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

StreamSegmentationInferenceRunner = RunSegmentationStreamInference
