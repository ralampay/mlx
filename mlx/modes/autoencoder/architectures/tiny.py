"""Small reference implementation of the vector autoencoder contract."""
from torch import nn


class TinyAutoencoder(nn.Module):
    def __init__(self, input_dimensions: int, bottleneck_dimensions: int) -> None:
        super().__init__()
        if not 1 <= bottleneck_dimensions < input_dimensions:
            raise ValueError("bottleneck_dimensions must be positive and smaller than input_dimensions")
        self.input_dimensions = input_dimensions
        self.bottleneck_dimensions = bottleneck_dimensions
        self.encoder = nn.Linear(input_dimensions, bottleneck_dimensions)
        self.decoder = nn.Linear(bottleneck_dimensions, input_dimensions)

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, embeddings):
        return self.decoder(embeddings)

    def forward(self, inputs):
        return self.decode(self.encode(inputs))


class TinyAutoencoderDefinition:
    name = "tiny"
    description = "Two linear layers demonstrating the vector autoencoder contract."

    def build(self, config):
        return TinyAutoencoder(int(config["input_dimensions"]), int(config["bottleneck_dimensions"]))
