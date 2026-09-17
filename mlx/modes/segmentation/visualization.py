from __future__ import annotations

import cv2
import numpy as np


def colorize_mask(mask: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    """Map class indices to an RGB palette without terminal or window side effects."""

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(palette):
        color_mask[mask == class_index] = color
    return color_mask


def blend_overlay(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend an RGB mask with an RGB image."""

    base = image_rgb.astype(np.uint8)
    return cv2.addWeighted(
        base,
        1.0 - alpha,
        color_mask.astype(np.uint8),
        alpha,
        0.0,
    )


def stack_segmentation_views(
    original_rgb: np.ndarray,
    predicted_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    *,
    palette: list[list[int]],
) -> np.ndarray:
    target_size = (original_rgb.shape[1], original_rgb.shape[0])
    color_mask = colorize_mask(predicted_mask, palette)
    color_mask = cv2.resize(color_mask, target_size, interpolation=cv2.INTER_NEAREST)
    overlay_rgb = cv2.resize(overlay_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    return np.hstack((original_bgr, mask_bgr, overlay_bgr))


def compose_segmentation_sample_panel(
    original_rgb: np.ndarray,
    ground_truth_rgb: np.ndarray,
    prediction_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
) -> np.ndarray:
    """Build a labeled RGB comparison panel for a qualitative test sample."""

    height, width = original_rgb.shape[:2]
    title_height = max(28, height // 12)
    panels = []
    for title, image in (
        ("Picture", original_rgb),
        ("Ground Truth", ground_truth_rgb),
        ("Prediction", prediction_rgb),
        ("Overlay", overlay_rgb),
    ):
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        titled = np.zeros((height + title_height, width, 3), dtype=np.uint8)
        titled[title_height:] = resized
        font_scale = max(0.4, min(0.8, width / 500.0))
        cv2.putText(
            titled,
            title,
            (8, title_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(titled)
    return np.hstack(panels)


__all__ = [
    "blend_overlay",
    "colorize_mask",
    "compose_segmentation_sample_panel",
    "stack_segmentation_views",
]
