"""Loading explicit component option objects; mode policy stays in its owner."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from mlx.core.exceptions import MLXUserError


def load_component_options(source: str | Mapping[str, Any] | None, *, purpose: str) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MLXUserError(f"Unable to read {purpose} configuration '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise MLXUserError(f"{purpose} configuration must be a JSON object.")
    return value
