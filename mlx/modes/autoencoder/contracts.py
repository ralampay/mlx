from __future__ import annotations
from typing import Any, Mapping, Protocol, runtime_checkable
import torch
from torch import nn
from mlx.core.exceptions import MLXUserError

@runtime_checkable
class VectorAutoencoder(Protocol):
    input_dimensions: int
    bottleneck_dimensions: int

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class AutoencoderDefinition(Protocol):
    name: str
    description: str

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        ...



def validate_model(model, *, input_dimensions: int, bottleneck_dimensions: int) -> None:
    if not isinstance(model, nn.Module):
        raise MLXUserError("Autoencoder definitions must build a torch.nn.Module.")
    if getattr(model, "input_dimensions", None) != input_dimensions:
        raise MLXUserError("Autoencoder model returned an unexpected input dimension.")
    if getattr(model, "bottleneck_dimensions", None) != bottleneck_dimensions:
        raise MLXUserError("Autoencoder model returned an unexpected bottleneck dimension.")
    if not callable(getattr(model, "encode", None)) or not callable(getattr(model, "decode", None)):
        raise MLXUserError("Autoencoder models must implement encode() and decode().")
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            probe = torch.zeros(2, input_dimensions)
            encoded = model.encode(probe)
            _shape(encoded, (2, bottleneck_dimensions), "encode")
            _shape(model.decode(encoded), probe.shape, "decode")
            _shape(model(probe), probe.shape, "forward")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise MLXUserError(f"Autoencoder model failed its shape probe: {exc}") from exc
    finally:
        for module, training in states:
            module.training = training


def _shape(value, expected, method):
    if not isinstance(value, torch.Tensor) or value.shape != expected or not torch.isfinite(value).all():
        raise MLXUserError(f"Autoencoder {method}() must return a finite tensor with shape {tuple(expected)}.")


def validate_loss(loss, *, training: bool) -> None:
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise MLXUserError("Autoencoder loss must return one finite scalar tensor.")
    if training and not loss.requires_grad:
        raise MLXUserError("Autoencoder training loss must retain gradients for backward().")
