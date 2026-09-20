from torch import nn


class AbsoluteError(nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean()


class AbsoluteErrorDefinition:
    name = "absolute-error"
    description = "A minimal scalar reconstruction loss."

    def build(self, config):
        if config:
            raise ValueError("absolute-error has no configuration options")
        return AbsoluteError()
