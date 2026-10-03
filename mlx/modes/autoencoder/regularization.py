"""Calibrated, batch-statistic regularizers (no sample-pair matching)."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError


class CalibratedReconstructionLoss(nn.Module):
    formula_version = 1
    minimum_batch_size = 2

    def __init__(self, rho=0.1, eta=1e-3):
        super().__init__()
        if isinstance(rho, bool) or not isinstance(rho, (float, int)) or not math.isfinite(rho) or rho < 0:
            raise MLXUserError("Regularization rho must be finite and nonnegative.")
        if isinstance(eta, bool) or not isinstance(eta, (float, int)) or not math.isfinite(eta) or eta <= 0:
            raise MLXUserError("Least Volume eta must be finite and positive.")
        self.rho, self.eta = float(rho), float(eta)
        self.requires_latent = self.rho > 0
        self.minimum_batch_size = 2 if self.requires_latent else 1
        self.scale = None if self.requires_latent else 0.0
        self.calibration = {}

    def penalty(self, latent):
        if latent is None or latent.ndim != 2 or latent.shape[0] < 2:
            raise MLXUserError("Regularization requires a matrix with at least two latent rows.")
        z = latent.float() if latent.dtype in (torch.float16, torch.bfloat16) else latent
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._penalty(z)

    def components(self, reconstruction, target, *, latent=None):
        rec = F.mse_loss(reconstruction, target)
        if not self.requires_latent:
            zero = rec.new_zeros(())
            return rec, {"reconstruction": rec, "penalty": zero, "weighted_penalty": zero}
        if self.scale is None:
            raise MLXUserError("Calibrate this regularizer on training rows before evaluating it.")
        penalty = self.penalty(latent)
        weighted = self.rho * self.scale * penalty
        return rec + weighted, {"reconstruction": rec, "penalty": penalty, "weighted_penalty": weighted}

    def forward(self, reconstruction, target, *, latent=None):
        return self.components(reconstruction, target, latent=latent)[0]

    @torch.no_grad()
    def calibrate(self, model, values):
        if not self.requires_latent:
            return
        training = model.training
        model.eval()
        try:
            latent = model.encode(values)
            rec = float(F.mse_loss(model.decode(latent), values))
            penalty = float(self.penalty(latent))
        finally:
            model.train(training)
        if not all(math.isfinite(v) and v > 0 for v in (rec, penalty)):
            raise MLXUserError("Regularizer calibration requires finite positive reconstruction and penalty values.")
        self.scale = rec / penalty
        if not math.isfinite(self.scale):
            raise MLXUserError("Regularizer calibration scale is nonfinite.")
        self.calibration = {"scale": self.scale, "reconstruction": rec, "penalty": penalty,
                            "rows": len(values), "formula_version": self.formula_version}


class LeastVolumeLoss(CalibratedReconstructionLoss):
    requires_constrained_decoder = True

    def _penalty(self, z):
        return (z.std(dim=0, correction=1) + self.eta).log().mean().exp()


class CovarianceLoss(CalibratedReconstructionLoss):
    def _penalty(self, z):
        if z.shape[1] < 2:
            raise MLXUserError("Covariance regularization requires at least two latent dimensions.")
        centered = z - z.mean(dim=0)
        covariance = centered.T @ centered / (len(z) - 1)
        mask = ~torch.eye(z.shape[1], device=z.device, dtype=torch.bool)
        return covariance[mask].square().mean()


class LeastVolumeLossDefinition:
    name = "mse-least-volume"
    description = "MSE plus calibrated Least Volume penalty; requires a constrained decoder."
    default_config = {"rho": 0.1, "eta": 1e-3}

    def build(self, config):
        if set(config) - set(self.default_config):
            raise MLXUserError("Least Volume accepts only rho and eta.")
        return LeastVolumeLoss(**config)


class CovarianceLossDefinition:
    name = "mse-covariance"
    description = "MSE plus calibrated mean squared off-diagonal feature covariance (paper-inspired)."
    default_config = {"rho": 0.1}

    def build(self, config):
        if set(config) - set(self.default_config):
            raise MLXUserError("Covariance loss accepts only rho.")
        return CovarianceLoss(**config)
