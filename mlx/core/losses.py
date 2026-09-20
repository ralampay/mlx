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


def validate_scalar_loss(loss, *, training: bool, context: str = "Training") -> None:
    """Check the shared scalar tensor contract, without owning an objective."""
    import torch
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise MLXUserError(f"{context} loss must return one finite scalar tensor.")
    if training and not loss.requires_grad:
        raise MLXUserError(f"{context} loss must retain gradients for backward().")
