from __future__ import annotations

from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import console


class RichTextEmbeddingReporter:
    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._task = None
        self._phase = None

    def emit(self, event: WorkflowEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        name = payload.get("event")
        if name in {"text_embedding_progress", "text_embedding_benchmark_progress"}:
            phase = payload.get("phase", "benchmark")
            if self._progress is None:
                self._progress = Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=console,
                )
                self._progress.start()
            if self._task is None or self._phase != phase:
                if self._task is not None:
                    self._progress.remove_task(self._task)
                self._phase = phase
                self._task = self._progress.add_task(str(phase).title(), total=event.total)
            self._progress.update(self._task, completed=event.current or 0)
        elif name in {"text_embedding_completed", "text_embedding_benchmark_completed"}:
            self._stop()
            console.print(Panel.fit(f"[bold green]{event.message}[/bold green]", border_style="green"))

    def _stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task = None
        self._phase = None


__all__ = ["RichTextEmbeddingReporter"]
