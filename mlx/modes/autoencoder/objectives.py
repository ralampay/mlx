"""Model-aware evaluation isolated from training orchestration."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError


@dataclass
class LossEvaluation:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]


class ReconstructionObjective:
    def __init__(self, criterion):
        self.criterion = criterion

    def evaluate(self, model, values, *, training):
        criterion = self.criterion
        if getattr(criterion, "requires_latent", False):
            latent = model.encode(values)
            prediction = model.decode(latent)
            if hasattr(criterion, "components"):
                loss, components = criterion.components(prediction, values, latent=latent)
                return LossEvaluation(loss, components)
            loss = criterion(prediction, values, latent=latent)
        else:
            prediction = model(values)
            if hasattr(criterion, "components"):
                loss, components = criterion.components(prediction, values)
                return LossEvaluation(loss, components)
            loss = criterion(prediction, values)
        return LossEvaluation(loss, {"reconstruction": F.mse_loss(prediction, values)})


class OrderedReconstructionObjective:
    def __init__(self, prefixes, seed):
        self.prefixes = prefixes
        self.generator = torch.Generator().manual_seed(seed)

    def evaluate(self, model, values, *, training):
        latent = model.encode(values)
        prefixes = self.prefixes
        if training:
            prefixes = (prefixes[int(torch.randint(len(prefixes), (1,), generator=self.generator))],)
        losses = []
        for width in prefixes:
            mask = torch.arange(latent.shape[1], device=latent.device) < width
            losses.append(F.mse_loss(model.decode(latent * mask), values))
        loss = torch.stack(losses).mean()
        return LossEvaluation(loss, {"reconstruction": loss})


def build_objective(model, criterion, seed):
    prefixes = getattr(model, "reconstruction_prefixes", None)
    if prefixes:
        if not isinstance(criterion, nn.MSELoss):
            raise MLXUserError("Ordered autoencoders currently support reconstruction MSE only.")
        return OrderedReconstructionObjective(prefixes, seed)
    if (getattr(criterion, "requires_constrained_decoder", False)
            and getattr(criterion, "requires_latent", False)
            and not getattr(model, "constrained_decoder", False)):
        raise MLXUserError("Least Volume requires a constrained decoder, such as simple-spectral.")
    return ReconstructionObjective(criterion)


def validate_training_variant(variant, request, width):
    """Construction-time capability validation; never calibrates or trains."""
    from mlx.modes.autoencoder.models import DEFAULT_AUTOENCODER_REGISTRY
    from mlx.modes.autoencoder.losses import DEFAULT_LOSS_REGISTRY
    reserved = {"input_dimensions", "hidden_dimensions", "bottleneck_dimensions"}
    if reserved & variant.model_config.keys():
        raise MLXUserError("Variant model_config cannot override dimension fields.")
    with torch.random.fork_rng(devices=[]):
        model = DEFAULT_AUTOENCODER_REGISTRY.resolve(variant.model)[0].build({
            **variant.model_config, "input_dimensions": max(768, width + 1),
            "hidden_dimensions": request.hidden_dim, "bottleneck_dimensions": width,
        })
        definition = DEFAULT_LOSS_REGISTRY.resolve(variant.loss)[0]
        criterion = definition.build({**getattr(definition, "default_config", {}), **variant.loss_config})
        build_objective(model, criterion, 0)
        for dimension in variant.evaluation_dimensions:
            if dimension != width and dimension not in getattr(model, "reconstruction_prefixes", ()):
                raise MLXUserError("Only ordered checkpoints can export a smaller prefix.")

        return getattr(model, "minimum_training_rows", 1)
