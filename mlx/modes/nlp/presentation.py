from __future__ import annotations

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import console


class RichEmbeddingReporter:
    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def emit(self, event: WorkflowEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        event_name = payload.get("event")
        if event_name == "embedding_started":
            console.print(f"[cyan]Rows to embed:[/cyan] {payload['rows']}")
            self._progress = Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "Embedding",
                total=int(payload["rows"]),
            )
        elif event_name == "embedding_progress":
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=event.current or 0,
                )
        elif event_name == "embedding_completed":
            self._stop_progress()
            console.print(
                Panel.fit(
                    "[bold green]Embedding export complete[/bold green]\n"
                    f"[cyan]File:[/cyan] {payload['output_path'].resolve()}\n"
                    f"[cyan]Rows:[/cyan] {payload['rows']}\n"
                    f"[cyan]Dimensions:[/cyan] {payload['dimensions']}",
                    border_style="green",
                )
            )
        elif event_name == "embedding_finished":
            self._stop_progress()

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
