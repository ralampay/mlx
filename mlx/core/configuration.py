"""Configuration boundary helpers; mode-specific defaults remain mode owned."""
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


def with_explicit_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy runner input, distinguishing parser defaults from Python-supplied values."""
    values = dict(config)
    explicit = config.get("_explicit_options")
    values["_explicit_options"] = (
        set(explicit) if explicit is not None else {
            name for name in config if not name.startswith("_")
        }
    )
    return values
