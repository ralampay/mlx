"""Explicit path resolution for adapter composition boundaries."""
from pathlib import Path
from typing import Any, Mapping

from mlx.core.exceptions import MLXUserError


def adapter_path(config: Mapping[str, Any], name: str, option: str, *, required: bool = True) -> Path | None:
    """Ignore unrelated parser defaults; Python mappings supply values directly."""
    explicit = config.get("_explicit_options")
    value = config.get(name) if explicit is None or name in explicit else None
    if value is None or not str(value).strip():
        if required:
            raise MLXUserError(f"Provide {option} explicitly for this adapter action.")
        return None
    return Path(value).expanduser().resolve()
