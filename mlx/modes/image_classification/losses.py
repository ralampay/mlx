"""image_classification loss catalog; aliases select scalar tensor losses."""
from types import MappingProxyType
from mlx.core.losses import build_scalar_loss

LOSS_DEFINITIONS = MappingProxyType({
    "cross-entropy": "torch.nn:CrossEntropyLoss",
    "bce": "torch.nn:BCELoss",
})


def build_loss(config, *, default="cross-entropy", entries=None):
    return build_scalar_loss(config.get("loss") or default,
                             LOSS_DEFINITIONS if entries is None else entries,
                             config.get("loss_config"))
