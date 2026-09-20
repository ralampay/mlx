"""Compatibility imports; checkpoint architecture paths remain stable."""
from mlx.modes.autoencoder.definitions import load_definition as _load_definition
from mlx.modes.autoencoder.contracts import AutoencoderDefinition, VectorAutoencoder
from mlx.modes.autoencoder.architectures.simple import SimpleAutoencoder, SimpleAutoencoderDefinition
from mlx.modes.autoencoder.model_registry import (
    AutoencoderRegistry, BUILTIN_AUTOENCODERS, DEFAULT_AUTOENCODER_REGISTRY,
)

__all__ = ["AutoencoderDefinition", "VectorAutoencoder", "SimpleAutoencoder",
           "SimpleAutoencoderDefinition", "AutoencoderRegistry", "DEFAULT_AUTOENCODER_REGISTRY"]
