from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.nlp.embedding import EmbedCsvCommand, EmbedCsvRequest


ACTION_HANDLERS = {
    "embed": lambda config: EmbedCsvCommand(
        EmbedCsvRequest.from_config({**config, "present": True})
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
