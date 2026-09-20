from __future__ import annotations

from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import console


class RichAutoencoderReporter:
    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._task = None

    def emit(self, event: WorkflowEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        name = payload.get("event")
        if name in {"autoencoder_training_epoch", "autoencoder_embedding_progress"}:
            if self._progress is None:
                self._progress = Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=console,
                )
                self._progress.start()
                description = "Training" if name == "autoencoder_training_epoch" else "Encoding"
                self._task = self._progress.add_task(description, total=event.total)
            self._progress.update(self._task, completed=event.current or 0)
        elif name in {"autoencoder_training_completed", "autoencoder_embedding_completed"}:
            self._stop()
            console.print(Panel.fit(f"[bold green]{event.message}[/bold green]", border_style="green"))

    def _stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task = None


def display_inventory(title: str, values) -> None:
    table = Table(title=title)
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for value in values:
        table.add_row(str(value["name"]), str(value["description"]))
    console.print(table)


__all__ = ["RichAutoencoderReporter", "display_inventory"]
