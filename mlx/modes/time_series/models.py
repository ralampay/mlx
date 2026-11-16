"""MLX PyTorch VAE implementations; TimeVAE-inspired additive decoder.

Independent implementation of the architecture family described by Desai et al.
(2021), https://arxiv.org/abs/2111.08095; not a bit-for-bit paper reproduction.
"""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError

from .requests import positive_int


def mlp(widths):
    layers = []
    for left, right in pairwise(widths):
        layers.extend((nn.Linear(left, right), nn.ReLU()))
    return nn.Sequential(*layers)


class TemporalVAE(nn.Module):
    """Shared Gaussian posterior, objective, and standard-normal prior contract."""

    def __init__(self, config, *, convolutional=False):
        super().__init__()
        self.length, self.features = config["window_size"], config["features"]
        self.latent_dim = config["latent_dim"]
        hidden = config["hidden_dims"]
        self.convolutional = convolutional
        if convolutional:
            layers, channels, length = [], self.features, self.length
            for width in hidden:
                layers.extend(
                    (nn.Conv1d(channels, width, 3, stride=2, padding=1), nn.ReLU())
                )
                channels, length = width, (length + 1) // 2
            self.encoder = nn.Sequential(*layers, nn.Flatten())
            encoded = channels * length
            self.encoded_shape = (channels, length)
            self.project = nn.Linear(self.latent_dim, encoded)
            layers = []
            channels = hidden[-1]
            for width in reversed(hidden[:-1]):
                layers.extend(
                    (
                        nn.ConvTranspose1d(channels, width, 4, stride=2, padding=1),
                        nn.ReLU(),
                    )
                )
                channels = width
            layers.append(
                nn.ConvTranspose1d(channels, self.features, 4, stride=2, padding=1)
            )
            self.decoder = nn.Sequential(*layers)
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(), mlp([self.length * self.features, *hidden])
            )
            encoded = hidden[-1]
            self.decoder = nn.Sequential(
                mlp([self.latent_dim, *reversed(hidden)]),
                nn.Linear(hidden[0], self.length * self.features),
            )
        self.mean = nn.Linear(encoded, self.latent_dim)
        self.logvar = nn.Linear(encoded, self.latent_dim)

    def encode(self, values):
        encoded = self.encoder(values.transpose(1, 2) if self.convolutional else values)
        return self.mean(encoded), self.logvar(encoded).clamp(-20, 20)

    def decode(self, latent):
        if self.convolutional:
            values = self.project(latent).reshape(-1, *self.encoded_shape)
            values = self.decoder(values)
            values = F.interpolate(
                values, size=self.length, mode="linear", align_corners=False
            )
            return values.transpose(1, 2)
        return self.decoder(latent).reshape(-1, self.length, self.features)

    def objective(self, values, kl_weight, generator):
        mean, logvar = self.encode(values)
        noise = torch.randn(mean.shape, generator=generator, device=mean.device)
        latent = mean + torch.exp(0.5 * logvar) * noise
        reconstruction = F.mse_loss(self.decode(latent), values)
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
        return {
            "loss": reconstruction + kl_weight * kl,
            "reconstruction": reconstruction,
            "kl": kl,
        }

    def sample(self, count, generator):
        return self.decode(
            torch.randn(
                count,
                self.latent_dim,
                generator=generator,
                device=next(self.parameters()).device,
            )
        )


class TimeVAE(TemporalVAE):
    def __init__(self, config):
        super().__init__(config, convolutional=True)
        options = config.get("model_config", {})
        unknown = set(options) - {"trend_degree", "seasonal_periods"}
        if unknown:
            raise MLXUserError(f"Unknown timevae model options: {sorted(unknown)}.")
        degree = positive_int(options.get("trend_degree", 2), "trend_degree", 0)
        periods = options.get("seasonal_periods", [])
        if not isinstance(periods, (list, tuple)):
            raise MLXUserError(
                "seasonal_periods must be a list of integer periods in timesteps."
            )
        for period in periods:
            positive_int(period, "seasonal_periods", 2)
        if (
            degree > 10
            or any(p > self.length for p in periods)
            or len(set(periods)) != len(periods)
        ):
            raise MLXUserError(
                "Use trend_degree <= 10 and distinct seasonal periods <= window-size."
            )
        self.level = nn.Linear(self.latent_dim, self.features)
        self.trend = (
            nn.Linear(self.latent_dim, degree * self.features) if degree else None
        )
        times = torch.linspace(-1, 1, self.length)
        self.register_buffer(
            "trend_basis",
            torch.stack([times**p for p in range(1, degree + 1)])
            if degree
            else torch.empty(0, self.length),
        )
        self.periods = tuple(periods)
        self.seasons = nn.ModuleList(
            nn.Linear(self.latent_dim, period * self.features) for period in periods
        )

    def decode(self, latent):
        result = super().decode(latent) + self.level(latent).unsqueeze(1)
        if self.trend is not None:
            coefficients = self.trend(latent).reshape(
                -1, len(self.trend_basis), self.features
            )
            result = result + torch.einsum(
                "bpf,pt->btf", coefficients, self.trend_basis
            )
        for period, layer in zip(self.periods, self.seasons):
            coefficients = layer(latent).reshape(-1, period, self.features)
            coefficients = coefficients - coefficients.mean(dim=1, keepdim=True)
            result = (
                result
                + coefficients[
                    :, torch.arange(self.length, device=latent.device) % period
                ]
            )
        return result


class VanillaDefinition:
    name = "vanilla-vae"
    description = "Dense window VAE; standard-normal prior, MSE + KL objective."

    def build(self, config):
        if config.get("model_config"):
            raise MLXUserError(
                "vanilla-vae has no model-specific options; use shared hyperparameters."
            )
        return TemporalVAE(config)


class ConvDefinition:
    name = "conv-vae"
    description = (
        "Temporal convolutional VAE; standard-normal prior, MSE + KL objective."
    )

    def build(self, config):
        if config.get("model_config"):
            raise MLXUserError(
                "conv-vae has no model-specific options; use shared hyperparameters."
            )
        return TemporalVAE(config, convolutional=True)


class TimeVAEDefinition:
    @property
    def default_config(self):
        return {"trend_degree": 2, "seasonal_periods": []}

    name = "timevae"
    description = "TimeVAE-inspired level/trend/seasonal/residual decoder; options: trend_degree=2, seasonal_periods=[]."

    def build(self, config):
        return TimeVAE(config)
