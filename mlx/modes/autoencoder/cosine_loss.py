"""Per-vector directional reconstruction without cross-sample similarities."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mlx.core.exceptions import MLXUserError


class MSECosineLoss(nn.Module):
    def __init__(self, cosine_weight=1.0):
        super().__init__()
        if (type(cosine_weight) not in (float, int) or not math.isfinite(cosine_weight)
                or cosine_weight < 0):
            raise MLXUserError("cosine_weight must be finite and nonnegative.")
        self.cosine_weight = float(cosine_weight)

    def components(self, reconstruction, target):
        if reconstruction.ndim != 2 or reconstruction.shape != target.shape or target.shape[1] < 1:
            raise MLXUserError("MSE-cosine requires matching two-dimensional reconstruction and target batches.")
        if reconstruction.dtype in (torch.float16, torch.bfloat16):
            reconstruction = reconstruction.float()
            target = target.float()
        mse = F.mse_loss(reconstruction, target)
        cosine = (1 - F.cosine_similarity(reconstruction, target, dim=1, eps=1e-8)).mean()
        weighted = cosine * (self.cosine_weight / target.shape[1])
        return mse + weighted, {"reconstruction": mse, "cosine": cosine, "weighted_cosine": weighted}

    def forward(self, reconstruction, target):
        return self.components(reconstruction, target)[0]


class MSECosineLossDefinition:
    name = "mse-cosine"
    description = "MSE plus input-dimension-scaled per-vector cosine reconstruction error."
    default_config = {"cosine_weight": 1.0}

    def build(self, config):
        if set(config) - {"cosine_weight"}:
            raise MLXUserError("mse-cosine accepts only cosine_weight.")
        return MSECosineLoss(config.get("cosine_weight", 1.0))
