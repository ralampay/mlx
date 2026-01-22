from pathlib import Path
from typing import Any, Dict

import typer

ModuleConfig = Dict[str, Any]


def _resolve_model_path(config: ModuleConfig) -> Path:
    model_value = config.get("model")
    if not model_value:
        raise typer.BadParameter("Local agents require --model pointing to a GGUF file.")

    model_path = Path(model_value)
    if not model_path.is_file():
        raise typer.BadParameter(f"Model file not found: {model_path}")

    if model_path.suffix.lower() != ".gguf":
        raise typer.BadParameter(
            "Local agent models must be GGUF files (.gguf)."
        )

    return model_path.resolve()


def _resolve_action(config: ModuleConfig) -> str:
    action_value = config.get("action")
    expected_action = "research-paper-summarizer"
    if action_value != expected_action:
        raise typer.BadParameter(
            f"Local agent action must be '{expected_action}'."
        )
    return action_value


def _print_config(model_path: Path, action: str) -> None:
    typer.secho("Running local agent with the following parameters:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Model : {model_path}")
    typer.echo("          (local GGUF model file)")
    typer.echo(f"  Action: {action}")


def run_agent(config: ModuleConfig):
    model_path = _resolve_model_path(config)
    action = _resolve_action(config)
    _print_config(model_path, action)
