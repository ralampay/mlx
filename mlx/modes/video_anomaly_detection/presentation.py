from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.presentation import RichInfrastructureEventRenderer
from mlx.core.ui import console, print_info, print_success, print_warning


def annotate_video_anomaly_frame(
    frame: np.ndarray,
    prediction: Mapping[str, Any] | None,
    frames_buffered: int,
    frames_required: int,
) -> np.ndarray:
    """Overlay the decision for the temporal window ending at this frame."""

    annotated = frame.copy()
    if prediction is None:
        color = (0, 191, 255)
        lines = (
            "WARMING UP",
            f"Temporal context: {frames_buffered}/{frames_required} frames",
        )
    else:
        is_anomaly = bool(prediction["is_anomaly"])
        color = (0, 0, 255) if is_anomaly else (0, 180, 0)
        lines = (
            "ANOMALY" if is_anomaly else "NORMAL",
            (
                f"Score: {float(prediction['anomaly_score']):.4f} | "
                f"Threshold: {float(prediction['threshold']):.4f}"
            ),
        )
    _draw_status_panel(annotated, lines=lines, color=color)
    return annotated


def _draw_status_panel(
    frame: np.ndarray,
    *,
    lines: tuple[str, ...],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scales = (0.8, 0.55)
    thicknesses = (2, 1)
    measurements = [
        cv2.getTextSize(line, font, scales[index], thicknesses[index])
        for index, line in enumerate(lines)
    ]
    panel_width = min(
        max(size[0][0] for size in measurements) + 24,
        frame.shape[1],
    )
    panel_height = min(68, frame.shape[0])
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for index, line in enumerate(lines):
        y = min(27 + index * 27, max(frame.shape[0] - 4, 0))
        cv2.putText(
            frame,
            line,
            (10, y),
            font,
            scales[index],
            color,
            thicknesses[index],
            cv2.LINE_AA,
        )


class _ProgressDisplay(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def add_task(self, description: str, **fields: Any) -> int:
        ...

    def update(self, task_id: int, **fields: Any) -> None:
        ...


class RichVideoAnomalyTrainingProgress:
    """Render each training phase as one transient, updating Rich line."""

    def __init__(
        self,
        progress_factory: Callable[[], _ProgressDisplay] | None = None,
    ) -> None:
        self._progress_factory = progress_factory or self._build_progress
        self._progress: _ProgressDisplay | None = None
        self._task_id: int | None = None

    def handle(self, event: WorkflowEvent) -> bool:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("event") != "video_anomaly_training_progress":
            return False

        status = str(payload.get("status") or "update")
        if status == "start":
            self.close()
            self._progress = self._progress_factory()
            self._task_id = self._progress.add_task(
                event.message,
                total=max(int(event.total or 0), 0),
                completed=max(int(event.current or 0), 0),
            )
            self._progress.start()
        elif self._progress is None:
            return True

        if self._progress is not None and self._task_id is not None:
            fields: dict[str, Any] = {
                "completed": max(int(event.current or 0), 0),
            }
            if event.total is not None:
                fields["total"] = max(int(event.total), 0)
            self._progress.update(self._task_id, **fields)

        if status in {"complete", "failed"}:
            self.close()
        return True

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task_id = None

    @staticmethod
    def _build_progress() -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            console=console,
            transient=True,
        )


class RichVideoAnomalyReporter:
    def __init__(
        self,
        infrastructure_events: RichInfrastructureEventRenderer | None = None,
        training_progress: RichVideoAnomalyTrainingProgress | None = None,
    ) -> None:
        self._infrastructure_events = (
            infrastructure_events or RichInfrastructureEventRenderer()
        )
        self._training_progress = training_progress or RichVideoAnomalyTrainingProgress()

    def emit(self, event: WorkflowEvent) -> None:
        if self._infrastructure_events.handle(event):
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        if self._training_progress.handle(event):
            return
        if payload.get("event") == "video_anomaly_training_epoch":
            metrics = payload["metrics"]
            console.print(
                f"[bold cyan]Epoch {event.current}/{event.total}[/bold cyan]  "
                f"train SVDD: [magenta]{float(metrics['train_svdd_loss']):.6f}[/magenta]  "
                f"validation: [magenta]{float(metrics['val_loss']):.6f}[/magenta]  "
                f"best: [green]{float(payload['best']):.6f}[/green]  "
                f"lr: {float(metrics['learning_rate']):.6g}"
            )
            return
        if payload.get("event") == "video_anomaly_benchmark":
            table = Table(title="Video Anomaly Benchmark")
            table.add_column("Level", style="cyan")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for level, metrics in payload["metrics"].items():
                for key in ("auroc", "auprc", "precision", "recall", "f1", "balanced_accuracy"):
                    table.add_row(level, key, f"{float(metrics[key]):.4f}")
            console.print(table)
            return
        if event.level == "success":
            print_success(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        else:
            print_info(event.message)


def display_video_anomaly_models(summaries) -> None:
    table = Table(title="Video Anomaly Detection Backbones", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("3D class")
    table.add_column("Drax fusion")
    table.add_column("Family")
    table.add_column("Feature dim", justify="right")
    table.add_column("Parameters", justify="right")
    table.add_column("Pretrained")
    table.add_column("Compatible")
    for item in summaries:
        table.add_row(
            item.model_name,
            item.backbone_class,
            item.drax_fusion_mode or "—",
            item.model_family,
            f"{item.feature_dim:,}",
            f"{item.parameter_count:,}",
            "yes" if item.pretrained_available else "no",
            "yes" if item.compatible else "no",
        )
    console.print(table)


__all__ = [
    "RichVideoAnomalyReporter",
    "RichVideoAnomalyTrainingProgress",
    "annotate_video_anomaly_frame",
    "display_video_anomaly_models",
]
