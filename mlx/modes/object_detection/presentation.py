from __future__ import annotations

import hashlib

import cv2
import numpy as np
from rich.panel import Panel
from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.presentation import RichInfrastructureEventRenderer
from mlx.core.ui import console, print_error, print_info, print_success, print_warning
from mlx.modes.object_detection.models import DetectionResult, ObjectDetectionBenchmarkResult


class RichWorkflowReporter:
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
        if isinstance(event.payload, dict):
            event_name = event.payload.get("event")
            if event_name == "training_summary":
                _print_key_value_panel(
                    "Training Configuration",
                    event.payload.get("values", {}),
                )
                return
            if event_name == "training_metrics":
                _print_rows_table(
                    "Final Validation Metrics",
                    event.payload.get("rows", ()),
                )
                return
            if event_name == "conversion_summary":
                _print_key_value_panel(
                    "Conversion Summary",
                    event.payload.get("values", {}),
                )
                return
        if event.level == "error":
            print_error(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        elif event.level == "success":
            print_success(event.message)
        else:
            print_info(event.message)


def _print_key_value_panel(title: str, values: object) -> None:
    if not isinstance(values, dict):
        print_info(str(values))
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="magenta")
    for key, value in values.items():
        table.add_row(str(key), str(value))
    console.print(Panel(table, title=title, border_style="cyan"))


def _print_rows_table(title: str, rows: object) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="magenta")
    if isinstance(rows, (list, tuple)):
        for row in rows:
            if isinstance(row, (list, tuple)) and len(row) == 2:
                table.add_row(str(row[0]), str(row[1]))
    console.print(table)


def print_benchmark_result(result: ObjectDetectionBenchmarkResult) -> None:
    table = Table(title="Object Detection Benchmark", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "map_50": "mAP@0.50",
        "map_50_95": "mAP@0.50:0.95",
    }
    for key, label in labels.items():
        table.add_row(label, f"{result.metrics[key]:.6f}")
    table.add_section()
    table.add_row("Provider", result.provider)
    table.add_row("Evaluator", result.evaluation_backend)
    table.add_row("Artifacts", str(result.output_dir))
    console.print(table)


def annotate_detections(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    annotated = frame.copy()
    for detection in result.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
        color = _color_for_label(detection.label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{detection.label}: {detection.confidence:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return tuple(int(min(max(int(digest[index : index + 2], 16), 64), 255)) for index in (0, 2, 4))
