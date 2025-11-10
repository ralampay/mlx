from __future__ import annotations

import os
from typing import Any, Callable, Dict

import typer
from rich import box
from rich.console import Console
from rich.table import Table

ModuleConfig = Dict[str, Any]

console = Console()

ENVIRONMENT_VARIABLES: Dict[str, Dict[str, Any]] = {
    "LOCAL_LLM_MODEL": {
        "description": "Filesystem path to the local language model weights.",
        "mask": False,
    },
    "LOCAL_LLM_GENERATION_MODEL": {
        "description": "Optional GGUF path used for local RAG answers.",
        "mask": False,
    },
    "OPENAI_API_KEY": {
        "description": "API key for OpenAI-powered modules.",
        "mask": True,
    },
    "HUGGINGFACE_TOKEN": {
        "description": "Access token for Hugging Face downloads and API calls.",
        "mask": True,
    },
    "DB_ADAPTER": {
        "description": "Vector database adapter (chromadb or postgres).",
        "mask": False,
    },
    "LLAMA_LOG_LEVEL": {
        "description": "Verbosity level for llama.cpp runtime logging.",
        "mask": False,
    },
    "DB_HOST": {
        "description": "Hostname for the ChromaDB instance when DB_ADAPTER=chromadb.",
        "mask": False,
    },
    "DB_PORT": {
        "description": "Port for the ChromaDB instance when DB_ADAPTER=chromadb.",
        "mask": False,
    },
    "DB_USERNAME": {
        "description": "Username for authenticated ChromaDB access.",
        "mask": False,
    },
    "DB_PASSWORD": {
        "description": "Password for authenticated ChromaDB access.",
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
        raise typer.BadParameter(
            f"Unsupported system action '{action}'. Available actions: {available}."
        )

    handler(config)
