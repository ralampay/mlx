from __future__ import annotations

from typing import Any

def run_nlp(config: dict[str, Any]) -> Any:
    """Compatibility entrypoint for callers importing the historic NLP runner."""

    from mlx.modes.text_embedding.runner import run_text_embedding

    return run_text_embedding(config)
