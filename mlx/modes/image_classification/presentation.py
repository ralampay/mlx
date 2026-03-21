from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.table import Table

from mlx.core.ui import console, print_info, print_success, print_warning


def print_config_summary(model: str, family: str, config: dict[str, Any]) -> None:
    table = Table(title=f"Configuration for {model} ({family})", show_lines=True)
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)


def _draw_header_bar(image, text: str):
    if image is None:
        return image

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.0, image.shape[1] / 250.0))
    thickness = max(1, int(image.shape[1] / 400))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    bar_height = text_height + baseline + 14
    bar = np.zeros((bar_height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (10, text_height + 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack((bar, image))


def display_similarity_matches(result: dict[str, Any]) -> None:
    input_image = result["input_image"]
    all_matches = result["top_matches"]
    best_label = result["best_match_label"]
    best_path = result["best_match_path"]
    best_distance = result["distance"]

    table = Table(title="Inference Results", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", justify="center", style="magenta")
    table.add_column("Image Path", justify="left")
    table.add_column("Distance", justify="center", style="green")

    for index, (label, path, distance) in enumerate(all_matches, start=1):
        table.add_row(str(index), label, str(path), f"{distance:.4f}")

    console.print(table)
    if best_label is not None:
        print_success(f"Best match: {best_label} (distance={best_distance:.4f})")

    images = []
    input_display = cv2.imread(str(input_image))
    if input_display is not None:
        images.append(_draw_header_bar(input_display, "INPUT"))

    for label, path, distance in all_matches:
        reference = cv2.imread(str(path))
        if reference is None:
            continue
        images.append(_draw_header_bar(reference, f"{label} - dist {distance:.4f}"))

    if not images:
        print_warning("No images to display.")
        return

    target_height = 200
    resized_images = []
    for image in images:
        height, width = image.shape[:2]
        scale = target_height / height
        resized_images.append(cv2.resize(image, (int(width * scale), target_height + 40)))

    num_cols = 5
    num_rows = math.ceil(len(resized_images) / num_cols)
    row_images = []
    for row_index in range(num_rows):
        row = resized_images[row_index * num_cols : (row_index + 1) * num_cols]
        while len(row) < num_cols:
            row.append(np.zeros_like(resized_images[0]))
        row_images.append(np.hstack(row))
    grid = np.vstack(row_images)
    cv2.imshow("Inference Comparison (All Samples)", grid)

    input_full = cv2.imread(str(input_image))
    best_full = cv2.imread(str(best_path)) if best_path else None
    if input_full is not None and best_full is not None:
        input_full = _draw_header_bar(input_full, "INPUT")
        best_full = _draw_header_bar(best_full, f"{best_label} - dist {best_distance:.4f}")

        input_height, input_width = input_full.shape[:2]
        best_height, best_width = best_full.shape[:2]
        target_height = min(400, max(input_height, best_height))
        input_resized = cv2.resize(
            input_full,
            (int(input_width * target_height / input_height), target_height),
        )
        best_resized = cv2.resize(
            best_full,
            (int(best_width * target_height / best_height), target_height),
        )
        cv2.imshow("Best Match Comparison", np.hstack((input_resized, best_resized)))

    print_info("Press any key on an image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_classification_predictions(result: dict[str, Any]) -> None:
    table = Table(title="Classification Predictions", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", style="magenta")
    table.add_column("Probability", justify="right", style="green")

    for index, (label, probability) in enumerate(result["top_predictions"], start=1):
        table.add_row(str(index), label, f"{probability:.4f}")

    console.print(table)
    if result.get("predicted_label"):
        print_success(f"Predicted label: {result['predicted_label']}")
