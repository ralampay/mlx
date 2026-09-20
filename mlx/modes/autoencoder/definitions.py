"""Construct mode-owned architecture and loss definitions from import references."""
from __future__ import annotations

import inspect

from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference


def load_definition(path: str, kind: str):
    value = load_reference(path, kind=kind)
    if not inspect.isclass(value):
        raise MLXUserError(f"{kind.title()} import '{path}' does not reference a class.")
    try:
        return value()
    except (TypeError, ValueError) as exc:
        raise MLXUserError(f"Unable to construct {kind} definition '{path}': {exc}") from exc

