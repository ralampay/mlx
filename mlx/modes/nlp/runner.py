from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.nlp.embedding import EmbedCsv


ACTION_HANDLERS = {
    "embed": lambda config: EmbedCsv(
        model_file=config.get("model_file"),
        input_file=config.get("input_file"),
        output_file=config.get("output_file"),
        column_name=config.get("column_name", "content"),
    ).execute(),
}


def run_nlp(config: dict[str, Any]) -> Any:
    action = config.get("action")
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for nlp. Available actions: {available}."
        )
    return handler(config)
