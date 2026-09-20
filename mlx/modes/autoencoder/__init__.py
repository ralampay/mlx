"""Generic one-dimensional vector autoencoder workflows."""

from mlx.modes.autoencoder.commands import EmbedAutoencoder, TrainAutoencoder
from mlx.modes.autoencoder.requests import AutoencoderEmbedRequest, AutoencoderTrainRequest

__all__ = [
    "AutoencoderEmbedRequest",
    "AutoencoderTrainRequest",
    "EmbedAutoencoder",
    "TrainAutoencoder",
]
