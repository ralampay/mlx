from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.table import Table

from mlx.core.ui import console, print_info, print_success, print_warning


def print_segmentation_config_summary(model: str, config: dict[str, Any]) -> None:
    table = Table(title=f"Configuration for {model} (segmentation)", show_lines=True)
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)


def colorize_mask(mask: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(palette):
        color_mask[mask == class_index] = color
    return color_mask


def blend_overlay(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    base = image_rgb.astype(np.uint8)
    overlay = cv2.addWeighted(base, 1.0 - alpha, color_mask.astype(np.uint8), alpha, 0.0)
    return overlay


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


def display_segmentation_result(result: dict[str, Any]) -> None:
    table = Table(title="Segmentation Inference", show_lines=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Input Image", str(result["input_image"]))
    table.add_row("Model", str(result["model_name"]))
    table.add_row("Input Size", f"{result['input_size'][0]}x{result['input_size'][1]}")
    table.add_row("Classes", str(result["num_classes"]))
    console.print(table)

    if result.get("window_image") is None:
        print_warning("No image window was created because the input image could not be rendered.")
        return

    print_success("Displaying original, predicted mask, and overlay.")
    print_info("Press any key on the image window to close...")
    cv2.imshow("MLX Segmentation", result["window_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

