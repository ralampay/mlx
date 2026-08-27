from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import cv2
import numpy as np
from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import (
    confirm_action,
    console,
    print_info,
    print_success,
    print_warning,
    prompt_float,
    prompt_int,
    prompt_text,
)
from mlx.modes.image_classification.requests import BuildImageClassificationDatasetRequest

if TYPE_CHECKING:
    from mlx.modes.image_classification.cam import CamResult


class RichImageClassificationReporter:
    def emit(self, event: WorkflowEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("event") == "benchmark_result":
            self._render_benchmark(payload)
            return
        if payload.get("event") == "classification_dataset_summary":
            table = Table(title="Label Summary", show_lines=True)
            table.add_column("Label", style="cyan")
            table.add_column("Images", justify="right", style="magenta")
            for label, count in payload["labels"].items():
                table.add_row(str(label), str(count))
            console.print(table)
            return
        if payload.get("event") == "tensor_output":
            print_success(event.message)
            print_info(f"Output tensor shape: {payload['shape']}")
            table = Table(title=str(payload["title"]), show_header=True)
            table.add_column("Index", justify="center", style="cyan")
            table.add_column("Value", justify="center", style="magenta")
            for index, value in enumerate(payload["values"]):
                table.add_row(str(index), f"{float(value):.6f}")
            console.print(table)
            return
        if event.level == "success":
            print_success(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        else:
            print_info(event.message)

    @staticmethod
    def _render_benchmark(payload: dict[str, Any]) -> None:
        metrics = payload["metrics"]
        table = Table(
            title=str(payload["title"]),
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Score", justify="right")
        for label, key in (
            ("Accuracy", "accuracy"),
            ("Ave Precision", "avg_precision"),
            ("Ave Recall", "avg_recall"),
            ("F1-score", "f1"),
            ("ROC AUC (macro)", "roc_auc_macro_ovr"),
            ("ROC AUC (weighted)", "roc_auc_weighted_ovr"),
            ("Average Precision", "average_precision"),
            ("Equal Error Rate", "equal_error_rate"),
            ("Best F1 Threshold", "best_f1_threshold"),
        ):
            if key in metrics:
                table.add_row(label, f"{float(metrics[key]):.4f}")
        console.print(table)

        class_names = payload.get("class_names") or []
        rows = []
        for class_name in class_names:
            slug = "".join(
                character.lower() if character.isalnum() else "_"
                for character in class_name
            ).strip("_")
            values = (
                metrics.get(f"auc_{slug}"),
                metrics.get(f"sensitivity_{slug}"),
                metrics.get(f"specificity_{slug}"),
            )
            if any(value is not None for value in values):
                rows.append((class_name, *values))
        if rows:
            class_table = Table(
                title="Per-Class Metrics",
                show_header=True,
                header_style="bold cyan",
            )
            class_table.add_column("Class", style="dim")
            class_table.add_column("AUC", justify="right")
            class_table.add_column("Sensitivity", justify="right")
            class_table.add_column("Specificity", justify="right")
            for class_name, auc_value, sensitivity, specificity in rows:
                class_table.add_row(
                    class_name,
                    f"{auc_value:.4f}" if auc_value is not None else "-",
                    f"{sensitivity:.4f}" if sensitivity is not None else "-",
                    f"{specificity:.4f}" if specificity is not None else "-",
                )
            console.print(class_table)


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
    best_similarity = result["similarity_score"]

    table = Table(title="Inference Results", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", justify="center", style="magenta")
    table.add_column("Image Path", justify="left")
    table.add_column("Same-class probability", justify="center", style="green")

    for index, (label, path, similarity) in enumerate(all_matches, start=1):
        table.add_row(str(index), label, str(path), f"{similarity:.4f}")

    console.print(table)
    if best_label is not None:
        print_success(f"Best match: {best_label} (same-class probability={best_similarity:.4f})")

    images = []
    input_display = cv2.imread(str(input_image))
    if input_display is not None:
        images.append(_draw_header_bar(input_display, "INPUT"))

    for label, path, similarity in all_matches:
        reference = cv2.imread(str(path))
        if reference is None:
            continue
        images.append(_draw_header_bar(reference, f"{label} - score {similarity:.4f}"))

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
        best_full = _draw_header_bar(best_full, f"{best_label} - score {best_similarity:.4f}")

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
    if result.get("accepted") is False:
        print_warning(
            "Image rejected as out-of-distribution "
            f"(score={result['ood_score']:.6f}, threshold={result['ood_threshold']:.6f})."
        )
        return
    table = Table(title="Classification Predictions", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", style="magenta")
    table.add_column("Probability", justify="right", style="green")

    for index, (label, probability) in enumerate(result["top_predictions"], start=1):
        table.add_row(str(index), label, f"{probability:.4f}")

    console.print(table)
    if result.get("predicted_label"):
        print_success(f"Predicted label: {result['predicted_label']}")


def display_cam_results(results: Iterable["CamResult"], *, delay: int = 0) -> None:
    for result in results:
        title = f"{result.method}: {result.source_path.name}"
        image_bgr = cv2.cvtColor(result.visualization, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, image_bgr)
        key = cv2.waitKey(delay)
        cv2.destroyWindow(title)
        if key in (ord("q"), 27):
            break


def resolve_image_dataset_build_request(
    request: BuildImageClassificationDatasetRequest,
    label_counts: dict[str, int],
) -> BuildImageClassificationDatasetRequest:
    del label_counts
    has_ratios = any(
        value is not None
        for value in (request.train_ratio, request.val_ratio, request.test_ratio)
    )
    split_mode = request.split_mode or ("ratios" if has_ratios else "counts")
    if split_mode == "ratios":
        resolved = replace(
            request,
            split_mode=split_mode,
            train_ratio=(
                request.train_ratio
                if request.train_ratio is not None
                else prompt_float("Train ratio?")
            ),
            val_ratio=(
                request.val_ratio
                if request.val_ratio is not None
                else prompt_float("Validation ratio?")
            ),
            test_ratio=(
                request.test_ratio
                if request.test_ratio is not None
                else prompt_float("Test ratio?")
            ),
        )
    else:
        resolved = replace(
            request,
            split_mode=split_mode,
            train_count=(
                request.train_count
                if request.train_count is not None
                else prompt_int("How many images per label for TRAIN?")
            ),
            val_count=(
                request.val_count
                if request.val_count is not None
                else prompt_int("How many images per label for VAL?")
            ),
            test_count=(
                request.test_count
                if request.test_count is not None
                else prompt_int("How many images per label for TEST?")
            ),
        )
    output_path = resolved.output_path or prompt_text(
        "Enter output path for split dataset"
    )
    overwrite = resolved.overwrite
    if Path(output_path).exists() and not overwrite:
        confirm_action(
            f"Output directory '{output_path}' already exists. Overwrite?",
            abort=True,
        )
        overwrite = True
    return replace(resolved, output_path=output_path, overwrite=overwrite)
