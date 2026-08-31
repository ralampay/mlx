from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.artifacts import load_image_one_class_checkpoint
from mlx.modes.image_recognition_oc.data import load_image_tensor
from mlx.modes.image_recognition_oc.requests import InferImageOneClassRequest


@dataclass(frozen=True)
class ImageOneClassInferenceResult:
    input_image: str
    model: str
    backbone: str
    predicted_label: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float


class InferImageOneClass:
    def __init__(
        self,
        request: InferImageOneClassRequest,
        *,
        reporter: WorkflowReporter | None = None,
        checkpoint_loader=load_image_one_class_checkpoint,
        image_loader=load_image_tensor,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.checkpoint_loader = checkpoint_loader
        self.image_loader = image_loader

    def execute(self) -> ImageOneClassInferenceResult:
        config = self.request.to_config()
        if not config.get("model_path"):
            raise MLXUserError("Image inference requires --model-path.")
        if not config.get("input_img"):
            raise MLXUserError("Image inference requires --input-img.")
        device = str(config["device"])
        model, checkpoint, stored, algorithm = self.checkpoint_loader(
            config["model_path"],
            device=device,
            model_name=config.get("model"),
            backbone_name=config.get("backbone"),
        )
        threshold = checkpoint.get("threshold")
        if threshold is None or not np.isfinite(float(threshold)):
            raise MLXUserError(
                "Image inference requires a checkpoint with a normal-validation calibrated threshold."
            )
        image = self.image_loader(
            config["input_img"],
            height=int(stored["height"]),
            width=int(stored["width"]),
            colored=bool(stored["colored"]),
        )
        model.eval()
        with torch.no_grad():
            score = float(algorithm.scores(model, image.unsqueeze(0).to(device))[0].item())
        if not np.isfinite(score):
            raise MLXUserError(
                "One-class image inference produced a non-finite anomaly score. "
                "Check the checkpoint and input image."
            )
        is_anomaly = score > float(threshold)
        result = ImageOneClassInferenceResult(
            input_image=str(Path(str(config["input_img"])).expanduser()),
            model=str(checkpoint["model_name"]),
            backbone=str(checkpoint["backbone_name"]),
            predicted_label="anomaly" if is_anomaly else "normal",
            is_anomaly=is_anomaly,
            anomaly_score=score,
            threshold=float(threshold),
        )
        emit(
            self.reporter,
            "warning" if is_anomaly else "success",
            f"Image classified as {result.predicted_label}: score={score:.6f}, threshold={float(threshold):.6f}",
            payload={"event": "image_one_class_inference", **result.__dict__},
        )
        return result


__all__ = ["ImageOneClassInferenceResult", "InferImageOneClass"]
