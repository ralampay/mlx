from __future__ import annotations

import cv2
import numpy as np


def probability_to_gray(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(probability) * 255.0), 0, 255).astype(np.uint8)


def probability_to_heatmap(probability: np.ndarray) -> np.ndarray:
    bgr = cv2.applyColorMap(probability_to_gray(probability), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def saliency_overlay(image_rgb: np.ndarray, probability: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = probability_to_heatmap(probability)
    return cv2.addWeighted(image_rgb.astype(np.uint8), 1.0 - alpha, heatmap, alpha, 0.0)


def compose_saliency_panel(
    original_rgb: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    overlay_rgb: np.ndarray,
) -> np.ndarray:
    height, width = original_rgb.shape[:2]
    title_height = max(28, height // 12)
    panels = []
    for title, image in (
        ("Original", original_rgb),
        ("Ground Truth", np.repeat(probability_to_gray(ground_truth)[..., None], 3, axis=2)),
        ("Predicted", np.repeat(probability_to_gray(prediction)[..., None], 3, axis=2)),
        ("Overlay", overlay_rgb),
    ):
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        panel = np.zeros((height + title_height, width, 3), dtype=np.uint8)
        panel[title_height:] = resized
        cv2.putText(
            panel,
            title,
            (8, title_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, min(0.8, width / 500.0)),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.hstack(panels)


def tensor_to_rgb(image) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(array, 3, axis=2) if array.shape[2] == 1 else array


def write_rgb(path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Unable to write image: {path}")


__all__ = [
    "compose_saliency_panel",
    "probability_to_gray",
    "probability_to_heatmap",
    "saliency_overlay",
    "tensor_to_rgb",
    "write_rgb",
]
