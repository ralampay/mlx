# Migrated from project-owned LibreYOLO adapters; retained MIT provenance.
"""Post-checkpoint adapter injection without changing foundation checkpoint keys."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .layers import LoRAConv2d
from .residual_fusion import DraxResidualFusionConv2d
from .hybrid import DraxHybridConv2d, DraxSpatialConv2d


class AdaptedFeature(nn.Module):
    def __init__(self, base: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        return self.adapter(self.base(*args, **kwargs))


@dataclass(frozen=True)
class InjectionReport:
    adapter: str
    modules: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int
    total_parameters: int
    trainable_percent: float


def count_parameters(model: nn.Module) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable,
            "trainable_percent": 100 * trainable / total if total else 0.0}


def inject_adapters(model: nn.Module, adapter: str, targets: dict[str, int], *,
                    reduction: int = 8, rank: int = 8, alpha: float = 1.0,
                    train_head: bool = False, trainable_modules: tuple[str, ...] = (),
                    registry=None) -> InjectionReport:
    """Validate and build all replacements before committing an injection.

    Factories may freeze their base parameters during construction. Original
    trainability is restored on failure, including failures at later targets.
    """
    from .registry import DEFAULT_FEATURE_ADAPTER_REGISTRY
    definition = (registry or DEFAULT_FEATURE_ADAPTER_REGISTRY).resolve(adapter)
    if not targets:
        raise ValueError("At least one adapter target is required")
    if reduction < 1 or rank < 1 or not math.isfinite(alpha):
        raise ValueError("reduction/rank must be positive and alpha finite")
    paths = tuple(targets)
    if any(not path for path in paths) or any(
        left.startswith(right + ".") for left in paths for right in paths if left != right
    ):
        raise ValueError("Adapter targets must be nonempty, non-overlapping module paths")
    additional = tuple(trainable_modules) + (("head",) if train_head else ())
    try:
        extra_parameters = [p for path in additional for p in model.get_submodule(path).parameters()]
        resolved = []
        seen = set()
        for path, channels in targets.items():
            parent_path, _, child_name = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            child = parent.get_submodule(child_name)
            if id(child) in seen:
                raise ValueError("Adapter targets must not alias the same module")
            seen.add(id(child))
            if getattr(child, "_mlx_feature_adapter", False) or isinstance(
                child, (AdaptedFeature, LoRAConv2d, DraxHybridConv2d, DraxSpatialConv2d, DraxResidualFusionConv2d)
            ):
                raise ValueError(f"Target {path} already has an adapter")
            definition.validate(child, channels)
            resolved.append((parent, child_name, child, channels))
    except AttributeError as exc:
        raise ValueError(f"Invalid adapter module path: {exc}") from exc
    original = [(p, p.requires_grad) for p in model.parameters()]
    replacements = []
    try:
        for parent, name, child, channels in resolved:
            target = child if definition.attachment == "conv" else channels
            replacement = definition.build(target, reduction=reduction, rank=rank, alpha=alpha)
            if definition.attachment == "feature":
                replacement = AdaptedFeature(child, replacement)
            reference = next(child.parameters(), None)
            if reference is None:
                reference = next(child.buffers(), None)
            if reference is not None:
                replacement.to(device=reference.device, dtype=reference.dtype if reference.is_floating_point() else None)
            replacement._mlx_feature_adapter = True
            replacements.append((parent, name, child, replacement))
        for parameter, _ in original:
            parameter.requires_grad_(False)
        for parent, name, child, replacement in replacements:
            setattr(parent, name, replacement)
        for parameter in extra_parameters:
            parameter.requires_grad_(True)
    except Exception:
        for parent, name, child, replacement in replacements:
            setattr(parent, name, child)
        for parameter, trainable in original:
            parameter.requires_grad_(trainable)
        raise
    counts = count_parameters(model)
    return InjectionReport(adapter, paths, counts["trainable"], counts["frozen"],
                           counts["total"], counts["trainable_percent"])


def adapter_state_dict(model: nn.Module) -> dict:
    """Trainable tensors only, for compact adapter-specific checkpoints."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable}


def load_adapter_state_dict(model: nn.Module, state: dict) -> None:
    """Restore a trainable-only state into the matching injected architecture."""
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(f"Adapter state mismatch: missing={sorted(expected - set(state))}, "
                         f"unexpected={sorted(set(state) - expected)}")
    for name, tensor in state.items():
        if parameters[name].shape != tensor.shape:
            raise ValueError(f"Adapter parameter {name} has shape {tuple(tensor.shape)}, "
                             f"expected {tuple(parameters[name].shape)}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor.to(parameters[name].device, parameters[name].dtype))
