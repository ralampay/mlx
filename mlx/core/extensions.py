"""Explicit import references shared by mode-owned extension factories.

This module stores no registrations and never constructs an implementation.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from mlx.core.exceptions import MLXUserError


def validate_reference(reference: str) -> tuple[str, str]:
    parts = reference.strip().split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Implementation references must use package.module:attribute.")
    module, attribute = parts
    if not all(part.isidentifier() for part in module.split(".")) or not attribute.isidentifier():
        raise ValueError("Implementation references must use package.module:attribute.")
    return module, attribute


def load_reference(reference: str, *, kind: str) -> Any:
    try:
        module, attribute = validate_reference(reference)
        return getattr(import_module(module), attribute)
    except (ImportError, AttributeError, ValueError) as exc:
        raise MLXUserError(f"Unable to load {kind} '{reference}': {exc}") from exc
