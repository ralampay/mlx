from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from mlx.core.exceptions import MLXAbort

console = Console()
stderr_console = Console(stderr=True)


def print_startup(mode: str, action: Optional[str], model: Optional[str]) -> None:
    summary = (
        f"[bold cyan]Mode[/bold cyan]: {mode}\n"
        f"[bold cyan]Action[/bold cyan]: {action or 'default'}\n"
        f"[bold cyan]Model[/bold cyan]: {model or 'default'}"
    )
    console.print(Panel.fit(summary, title="MLX", border_style="cyan"))


def print_info(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


def print_success(message: str) -> None:
    console.print(f"[bold green]{message}[/bold green]")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]{message}[/bold yellow]")


def print_error(message: str) -> None:
    stderr_console.print(Panel.fit(message, title="Error", border_style="red"))


def prompt_text(message: str, default: Optional[str] = None) -> str:
    if default is None:
        return Prompt.ask(f"[bold green]{message}[/bold green]")
    return Prompt.ask(f"[bold green]{message}[/bold green]", default=default)


def prompt_int(message: str, default: Optional[int] = None) -> int:
    if default is None:
        return IntPrompt.ask(f"[bold green]{message}[/bold green]")
    return IntPrompt.ask(f"[bold green]{message}[/bold green]", default=default)


def prompt_float(message: str, default: Optional[float] = None) -> float:
    if default is None:
        return FloatPrompt.ask(f"[bold green]{message}[/bold green]")
    return FloatPrompt.ask(f"[bold green]{message}[/bold green]", default=default)


def confirm_action(message: str, default: bool = False, abort: bool = False) -> bool:
    confirmed = Confirm.ask(f"[bold yellow]{message}[/bold yellow]", default=default)
    if abort and not confirmed:
        raise MLXAbort()
    return confirmed
