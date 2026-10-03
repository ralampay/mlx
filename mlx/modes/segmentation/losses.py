"""segmentation loss catalog; aliases select scalar tensor losses."""
from types import MappingProxyType
import torch
from torch import nn
from torch.nn import functional as F
from mlx.core.losses import build_scalar_loss


class CrossEntropyDiceLoss(nn.Module):
    """Cross entropy with foreground soft Dice for sparse binary masks."""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0,
                 foreground_weight: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError("dice_weight must be between 0 and 1.")
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        if foreground_weight <= 0:
            raise ValueError("foreground_weight must be positive.")
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer("class_weights", torch.tensor([1.0, foreground_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != 2:
            raise ValueError("CrossEntropyDiceLoss requires two output classes.")
        ce = F.cross_entropy(logits, targets, weight=self.class_weights)
        foreground = logits.softmax(dim=1)[:, 1]
        truth = (targets == 1).to(foreground.dtype)
        intersection = (foreground * truth).sum(dim=(-2, -1))
        denominator = foreground.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return (1 - self.dice_weight) * ce + self.dice_weight * dice_loss


LOSS_DEFINITIONS = MappingProxyType({
    "cross-entropy": "torch.nn:CrossEntropyLoss",
    "cross-entropy-dice": "mlx.modes.segmentation.losses:CrossEntropyDiceLoss",
})


def build_loss(config, *, default="cross-entropy", entries=None):
    return build_scalar_loss(config.get("loss") or default,
                             LOSS_DEFINITIONS if entries is None else entries,
                             config.get("loss_config"))
