"""Construction and validation shared by native scalar-loss factories.

Catalogs and target semantics remain mode-owned.
"""
from __future__ import annotations
from typing import Mapping
from mlx.core.configuration import load_component_options
from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference


def build_scalar_loss(name: str, entries: Mapping[str, str], options=None):
    from torch import nn
    reference = name if ":" in name else entries.get(name)
    if reference is None:
        raise MLXUserError(f"Unsupported loss '{name}'. Available: {', '.join(sorted(entries))}.")
    definition = load_reference(reference, kind="loss")
    config = load_component_options(options, purpose="loss")
    try:
        loss = definition().build(config) if hasattr(definition, "build") else definition(**config)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise MLXUserError(f"Cannot construct loss '{name}': {exc}") from exc
    if not isinstance(loss, nn.Module):
        raise MLXUserError(f"Loss '{name}' must construct a torch.nn.Module.")
    return loss
