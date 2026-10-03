"""Structured variants of the existing vector MLP."""
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.architectures.simple import SimpleAutoencoderDefinition


class SpectralAutoencoderDefinition:
    name = "simple-spectral"
    description = "GELU MLP with spectral normalization on decoder linear layers."

    def build(self, config):
        model = SimpleAutoencoderDefinition().build(config)
        # Parametrization initialization must not perturb subsequent random streams.
        with torch.random.fork_rng(devices=[]):
            for layer in model.decoder:
                if isinstance(layer, nn.Linear):
                    spectral_norm(layer, n_power_iterations=5)
        model.constrained_decoder = True
        return model


class OrderedAutoencoderDefinition:
    name = "ordered-simple"
    description = "GELU MLP trained to reconstruct from ordered latent prefixes."

    def build(self, config):
        model = SimpleAutoencoderDefinition().build(config)
        prefixes = config.get("prefix_dimensions", [128, 256, 384, 512])
        if (not isinstance(prefixes, (list, tuple)) or not prefixes
                or any(type(d) is not int or not 1 <= d <= model.bottleneck_dimensions for d in prefixes)
                or len(set(prefixes)) != len(prefixes) or max(prefixes) != model.bottleneck_dimensions):
            raise MLXUserError("Ordered prefix_dimensions must be unique positive widths including the full bottleneck.")
        model.reconstruction_prefixes = tuple(sorted(prefixes))
        return model
