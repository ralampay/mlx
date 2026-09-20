from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def ssim_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM loss for single-channel saliency maps."""

    if probabilities.shape != targets.shape:
        raise ValueError("SSIM inputs must have matching shapes.")
    kernel_size = min(11, probabilities.shape[-2], probabilities.shape[-1])
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(1, kernel_size)
    padding = kernel_size // 2
    mean_x = F.avg_pool2d(probabilities, kernel_size, stride=1, padding=padding)
    mean_y = F.avg_pool2d(targets, kernel_size, stride=1, padding=padding)
    variance_x = F.avg_pool2d(probabilities.square(), kernel_size, 1, padding) - mean_x.square()
    variance_y = F.avg_pool2d(targets.square(), kernel_size, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(probabilities * targets, kernel_size, 1, padding) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1)
        * (variance_x + variance_y + c2)
    ).clamp_min(1e-12)
    return 1.0 - score.mean()


def iou_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    union = (probabilities + targets - probabilities * targets).sum(dim=dimensions)
    return (1.0 - (intersection + 1.0) / (union + 1.0)).mean()


@dataclass(frozen=True)
class SaliencyLossResult:
    loss: torch.Tensor
    bce_loss: torch.Tensor
    ssim_loss: torch.Tensor
    iou_loss: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "loss": self.loss,
            "bce_loss": self.bce_loss,
            "ssim_loss": self.ssim_loss,
            "iou_loss": self.iou_loss,
        }

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.as_dict()[name]

    def items(self):
        return self.as_dict().items()


class SaliencyHybridLoss(nn.Module):
    def __init__(self, *, bce_weight: float = 1.0, ssim_weight: float = 1.0, iou_weight: float = 1.0) -> None:
        super().__init__()
        if min(bce_weight, ssim_weight, iou_weight) < 0:
            raise ValueError("Saliency loss weights must be non-negative.")
        if bce_weight + ssim_weight + iou_weight == 0:
            raise ValueError("At least one saliency loss weight must be positive.")
        self.bce_weight = float(bce_weight)
        self.ssim_weight = float(ssim_weight)
        self.iou_weight = float(iou_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> SaliencyLossResult:
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Saliency logits and targets must both have shape [B, 1, H, W].")
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        ssim = ssim_loss(probabilities, targets)
        iou = iou_loss(probabilities, targets)
        total = self.bce_weight * bce + self.ssim_weight * ssim + self.iou_weight * iou
        return SaliencyLossResult(total, bce, ssim, iou)


__all__ = ["SaliencyHybridLoss", "SaliencyLossResult", "iou_loss", "ssim_loss"]
