from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.commands import NullWorkflowReporter
from mlx.modes.nlp.embedding import EmbedCsvCommand, EmbedCsvRequest
from mlx.modes.nlp.presentation import RichEmbeddingReporter


def _embed(config: dict[str, Any]):
    is_json = config.get("output_format") == "json"
    return EmbedCsvCommand(
        EmbedCsvRequest.from_config({**config, "present": not is_json}),
        reporter=NullWorkflowReporter() if is_json else RichEmbeddingReporter(),
    ).execute()


ACTION_HANDLERS = {"embed": _embed}


def run_nlp(config: dict[str, Any]) -> Any:
    action = config.get("action") or "embed"
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for nlp. Available actions: {available}."
        )
    return handler(config)
