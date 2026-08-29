from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

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


__all__ = ["RichDatasetDownloadProgress", "RichInfrastructureEventRenderer"]
