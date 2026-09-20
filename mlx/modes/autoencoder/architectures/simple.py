from __future__ import annotations
from typing import Any, Mapping
from torch import nn
from mlx.core.exceptions import MLXUserError

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dimensions: int, hidden_dimensions: int, bottleneck_dimensions: int) -> None:
        super().__init__()
        if input_dimensions < 2:
            raise ValueError("input_dimensions must be at least 2")
        if not 1 <= bottleneck_dimensions < input_dimensions:
            raise ValueError("bottleneck_dimensions must be positive and smaller than input_dimensions")
        if hidden_dimensions < bottleneck_dimensions:
            raise ValueError("hidden_dimensions must be at least bottleneck_dimensions")
        self.input_dimensions = int(input_dimensions)
        self.hidden_dimensions = int(hidden_dimensions)
        self.bottleneck_dimensions = int(bottleneck_dimensions)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dimensions, self.hidden_dimensions),
            nn.GELU(),
            nn.Linear(self.hidden_dimensions, self.bottleneck_dimensions),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dimensions, self.hidden_dimensions),
            nn.GELU(),
            nn.Linear(self.hidden_dimensions, self.input_dimensions),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder(embeddings)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


class SimpleAutoencoderDefinition:
    name = "simple"
    description = "Symmetric GELU MLP with a linear bottleneck and reconstruction output."

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        try:
            return SimpleAutoencoder(
                input_dimensions=int(config["input_dimensions"]),
                hidden_dimensions=int(config.get("hidden_dimensions", 256)),
                bottleneck_dimensions=int(config.get("bottleneck_dimensions", 128)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MLXUserError(f"Invalid simple autoencoder configuration: {exc}") from exc
