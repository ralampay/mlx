from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import console


class _ProgressDisplay(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def add_task(self, description: str, **fields: Any) -> int:
        ...

    def update(self, task_id: int, **fields: Any) -> None:
        ...


class RichDatasetDownloadProgress:
    """Render shared dataset-download events as one updating terminal line."""

    def __init__(
        self,
        progress_factory: Callable[[], _ProgressDisplay] | None = None,
    ) -> None:
        self._progress_factory = progress_factory or self._build_progress
        self._progress: _ProgressDisplay | None = None
        self._task_id: int | None = None

    def handle(self, event: WorkflowEvent) -> bool:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("event") != "dataset_download":
            return False

        status = str(payload.get("status") or "update")
        if self._progress is None:
            self._progress = self._progress_factory()
            self._task_id = self._progress.add_task(
                event.message,
                total=max(int(event.total or 0), 0),
                completed=max(int(event.current or 0), 0),
            )
            self._progress.start()

        if self._task_id is not None:
            fields: dict[str, Any] = {
                "completed": max(int(event.current or 0), 0),
            }
            if event.total is not None:
                fields["total"] = max(int(event.total), 0)
            if status == "complete":
                fields["description"] = "[green]S3 dataset downloaded[/green]"
            elif status == "failed":
                fields["description"] = "[red]S3 dataset download interrupted[/red]"
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
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeRemainingColumn(compact=True),
            console=console,
            transient=False,
        )


class RichInfrastructureEventRenderer:
    """Render shared infrastructure events for composition by mode reporters."""

    def __init__(
        self,
        dataset_download_progress: RichDatasetDownloadProgress | None = None,
    ) -> None:
        self._dataset_download_progress = (
            dataset_download_progress or RichDatasetDownloadProgress()
        )

    def handle(self, event: WorkflowEvent) -> bool:
        return self._dataset_download_progress.handle(event)


@dataclass(frozen=True)
class TrainingMetricSpec:
    """Presentation metadata for one value in a training-epoch event."""

    key: str
    label: str
    higher_is_better: bool | None
    precision: int = 4


class RichTrainingMetricsRenderer:
    """Render persistent, comparable metric rows from structured epoch events."""

    def __init__(
        self,
        metrics: tuple[TrainingMetricSpec, ...],
        *,
        event_names: tuple[str, ...] = ("training_epoch",),
        highlights: tuple[tuple[str, str], ...] = (),
        render: Callable[[RenderableType], None] | None = None,
    ) -> None:
        self._metrics = metrics
        self._event_names = frozenset(event_names)
        self._highlights = highlights
        self._render = render or console.print
        self._previous_metrics: dict[str, float] = {}
        self._last_epoch: int | None = None

    def handle(self, event: WorkflowEvent) -> bool:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("event") not in self._event_names:
            return False

        current = int(event.current or 0)
        total = int(event.total or 0)
        raw_metrics = payload.get("metrics")
        current_metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
        if self._last_epoch is not None and current <= self._last_epoch:
            self.reset()

        raw_previous = payload.get("previous_metrics")
        previous_metrics = (
            dict(raw_previous)
            if isinstance(raw_previous, dict)
            else self._previous_metrics
        )
        line = Text()
        line.append(f"Epoch {current}/{total}", style="bold cyan")
        for spec in self._metrics:
            value = _finite_float(current_metrics.get(spec.key))
            if value is None:
                continue
            line.append(f"  {spec.label} ", style="dim")
            line.append(f"{value:.{spec.precision}f}", style="magenta")
            previous = _finite_float(previous_metrics.get(spec.key))
            if previous is None:
                line.append(" —", style="dim")
            else:
                delta = value - previous
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                if spec.higher_is_better is None:
                    style = "dim"
                else:
                    improved = delta > 0 if spec.higher_is_better else delta < 0
                    style = "green" if improved else "red" if delta else "dim"
                line.append(
                    f" {arrow}{abs(delta):.{spec.precision}f}",
                    style=style,
                )

        for key, label in self._highlights:
            if payload.get(key):
                line.append(f"  {label}", style="bold green")
        self._render(line)
        self._previous_metrics = {
            key: value
            for key, raw_value in current_metrics.items()
            if (value := _finite_float(raw_value)) is not None
        }
        self._last_epoch = current
        return True

    def reset(self) -> None:
        self._previous_metrics = {}
        self._last_epoch = None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


__all__ = [
    "RichDatasetDownloadProgress",
    "RichInfrastructureEventRenderer",
    "RichTrainingMetricsRenderer",
    "TrainingMetricSpec",
]
