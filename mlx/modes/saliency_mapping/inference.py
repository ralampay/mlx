from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.saliency_mapping.checkpoints import load_checkpoint_bundle
from mlx.modes.saliency_mapping.data import load_saliency_image_tensor
from mlx.modes.saliency_mapping.requests import SaliencyRequest
from mlx.modes.saliency_mapping.visualization import (
    probability_to_gray,
    probability_to_heatmap,
    saliency_overlay,
    write_rgb,
)


class InferSaliencyImage:
    def __init__(self, request: SaliencyRequest, *, model_registry=None) -> None:
        self.model_registry = model_registry
        self.request = request

    def execute(self) -> dict[str, Any]:
        config = self.request.to_config()
        model, metadata = load_checkpoint_bundle(config, **({"model_registry": self.model_registry} if self.model_registry is not None else {}))
        device = self.request.device
        model = model.to(device).eval()
        input_path = Path(self.request.input_img).expanduser()
        if not input_path.is_file():
            raise MLXUserError(f"Input image not found: {input_path}")
        tensor = load_saliency_image_tensor(
            input_path,
            input_size=tuple(metadata["input_size"]),
            colored=bool(metadata["colored"]),
            transform=str(metadata["transform"]),
        )
        with torch.inference_mode():
            logits = model(tensor.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        original_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise MLXUserError(f"Unable to read input image: {input_path}")
        original = cv2.cvtColor(
            cv2.resize(original_bgr, tuple(metadata["input_size"]), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        output = Path(
            self.request.output_path or input_path.with_name(f"{input_path.stem}-saliency")
        ).expanduser()
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MLXUserError(
                f"Unable to create saliency inference output directory '{output}': {exc}"
            ) from exc
        probability_npy = output / "probability.npy"
        probability_png = output / "probability.png"
        heatmap_path = output / "heatmap.png"
        overlay_path = output / "overlay.png"
        try:
            np.save(probability_npy, probability)
            if not cv2.imwrite(str(probability_png), probability_to_gray(probability)):
                raise OSError(f"Unable to write image: {probability_png}")
            write_rgb(heatmap_path, probability_to_heatmap(probability))
            write_rgb(
                overlay_path,
                saliency_overlay(original, probability, self.request.overlay_alpha),
            )
        except OSError as exc:
            raise MLXUserError(f"Unable to write saliency inference artifacts: {exc}") from exc
        return {
            "input_image": input_path,
            "model_name": metadata["model_name"],
            "input_size": metadata["input_size"],
            "probability_map": probability,
            "probability_npy": probability_npy,
            "probability_png": probability_png,
            "heatmap": heatmap_path,
            "overlay": overlay_path,
        }


__all__ = ["InferSaliencyImage"]
