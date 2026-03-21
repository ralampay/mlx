from __future__ import annotations

import os
from typing import Any, Callable, Dict

from rich import box
from rich.console import Console
from rich.table import Table
from mlx.core.exceptions import MLXUserError

ModuleConfig = Dict[str, Any]

console = Console()

ENVIRONMENT_VARIABLES: Dict[str, Dict[str, Any]] = {
    "ROBOFLOW_API_KEY": {
        "description": "Optional API key for Roboflow dataset workflows.",
        "mask": True,
    },
}


def _mask_value(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _render_environment_table(_: ModuleConfig) -> None:
    table = Table(title="MLX Environment Variables", box=box.SIMPLE_HEAVY)
    table.add_column("Variable", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="bold green")
    table.add_column("Description", style="white")

    for name, spec in ENVIRONMENT_VARIABLES.items():
        raw_value = os.environ.get(name)
        if raw_value:
            value_display = raw_value if not spec["mask"] else _mask_value(raw_value)
        else:
            value_display = "[red]not set[/]"
        table.add_row(name, value_display, spec["description"])

    console.print(table)

ACTION_HANDLERS: Dict[str, Callable[[ModuleConfig], None]] = {
    "ls-env": _render_environment_table,
}


def run_system(config: ModuleConfig) -> None:
    action = config.get("action")
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported system action '{action}'. Available actions: {available}."
        )

    handler(config)
