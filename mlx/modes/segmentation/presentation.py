from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.presentation import RichInfrastructureEventRenderer
from mlx.core.ui import (
    confirm_action,
    console,
    print_info,
    print_success,
    print_warning,
    prompt_int,
    prompt_text,
)
from mlx.modes.segmentation.requests import BuildSegmentationDatasetRequest
from mlx.modes.segmentation.visualization import (
    blend_overlay,
    colorize_mask,
    stack_segmentation_views,
)


class RichSegmentationReporter:
    def __init__(
        self,
        infrastructure_events: RichInfrastructureEventRenderer | None = None,
    ) -> None:
        self._infrastructure_events = (
            infrastructure_events or RichInfrastructureEventRenderer()
        )

    def emit(self, event: WorkflowEvent) -> None:
        if self._infrastructure_events.handle(event):
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        event_name = payload.get("event")
        if event_name == "segmentation_tensor_output":
            print_success(event.message)
            print_info(f"Output tensor shape: {payload['shape']}")
            table = Table(title="Segmentation Model Output", show_header=True)
            table.add_column("Index", justify="center", style="cyan")
            table.add_column("Value", justify="center", style="magenta")
            for index, value in enumerate(payload["values"]):
                table.add_row(str(index), f"{float(value):.6f}")
            console.print(table)
            return
        if event_name == "segmentation_epoch":
            metrics = payload["metrics"]
            table = Table(
                title=f"Epoch {event.current}/{event.total}",
                show_lines=True,
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="magenta")
            for label, value in (
                ("Train Loss", payload["train_loss"]),
                ("Validation Loss", payload["val_loss"]),
                ("Pixel Accuracy", metrics["pixel_accuracy"]),
                ("Macro Dice", metrics["macro_dice"]),
                ("Foreground Dice", metrics["mean_foreground_dice"]),
                ("Mean IoU", metrics["mean_iou"]),
                ("Foreground IoU", metrics["mean_foreground_iou"]),
            ):
                table.add_row(label, f"{float(value):.6f}")
            console.print(table)
            for checkpoint in payload["checkpoints"]:
                print_success(checkpoint)
            return
        if event_name == "segmentation_benchmark":
            metrics = payload["metrics"]
            table = Table(title="Segmentation Benchmark Results", show_lines=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="magenta")
            for key in (
                "cross_entropy_loss",
                "pixel_accuracy",
                "macro_dice",
                "mean_foreground_dice",
                "mean_iou",
                "mean_foreground_iou",
                "frequency_weighted_iou",
                "cohen_kappa",
                "multiclass_mcc",
                "macro_roc_auc",
                "macro_average_precision",
                "expected_calibration_error",
                "images_per_second_forward",
            ):
                table.add_row(key, f"{float(metrics[key]):.6f}")
            console.print(table)
            class_table = Table(title="Per-Class Segmentation Metrics")
            for heading in (
                "Class",
                "Precision",
                "Recall",
                "Specificity",
                "Dice",
                "IoU",
                "ROC AUC",
            ):
                class_table.add_column(
                    heading,
                    justify="right" if heading != "Class" else "left",
                )
            for row in payload["class_rows"]:
                class_table.add_row(
                    str(row["class_name"]),
                    *(
                        _format_metric(row.get(key))
                        for key in (
                            "precision",
                            "recall",
                            "specificity",
                            "dice",
                            "iou",
                            "roc_auc",
                        )
                    ),
                )
            console.print(class_table)
            return
        if event_name == "segmentation_dataset_summary":
            table = Table(title="Segmentation Pair Summary", show_lines=True)
            table.add_column("Directory", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Images Dir", str(payload["images_dir"]))
            table.add_row("Masks Dir", str(payload["masks_dir"]))
            table.add_row("Pairs", str(payload["pairs"]))
            console.print(table)
            return
        if event.level == "success":
            print_success(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        else:
            print_info(event.message)


def print_segmentation_config_summary(model: str, config: dict[str, Any]) -> None:
    table = Table(title=f"Configuration for {model} (segmentation)", show_lines=True)
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)


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


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    return f"{numeric:.4f}" if np.isfinite(numeric) else "nan"


def resolve_segmentation_dataset_build_request(
    request: BuildSegmentationDatasetRequest,
    pair_count: int,
) -> BuildSegmentationDatasetRequest:
    del pair_count
    resolved = replace(
        request,
        train_count=(
            request.train_count
            if request.train_count is not None
            else prompt_int("How many paired samples for TRAIN?")
        ),
        val_count=(
            request.val_count
            if request.val_count is not None
            else prompt_int("How many paired samples for VAL?")
        ),
        test_count=(
            request.test_count
            if request.test_count is not None
            else prompt_int("How many paired samples for TEST?")
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
