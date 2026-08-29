from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mlx.core.exceptions import MLXUserError
from mlx.core.deep_svdd import quantile_threshold
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier


def validate_svdd_config(config: dict) -> None:
    method = config.get("ood_method", "none")
    if method not in {"none", "deep-svdd"}:
        raise MLXUserError("--ood-method must be one of: none, deep-svdd.")
    if method == "none":
        return
    if float(config.get("svdd_weight", 0.05)) < 0:
        raise MLXUserError("--svdd-weight must be greater than or equal to zero.")
    if int(config.get("svdd_dim", 128)) <= 0:
        raise MLXUserError("--svdd-dim must be greater than zero.")
    if int(config.get("svdd_hidden_dim", 256)) <= 0:
        raise MLXUserError("--svdd-hidden-dim must be greater than zero.")
    quantile = float(config.get("svdd_quantile", 0.95))
    if not 0 < quantile < 1:
        raise MLXUserError("--svdd-quantile must be strictly between zero and one.")
    if int(config.get("svdd_warmup_epochs", 0)) < 0:
        raise MLXUserError("--svdd-warmup-epochs must be greater than or equal to zero.")


def compute_svdd_loss(model: JointDeepSVDDClassifier, embedding: torch.Tensor) -> torch.Tensor:
    return model.compute_svdd_score(embedding).mean()


@torch.no_grad()
def initialize_svdd_center(
    model: JointDeepSVDDClassifier,
    loader: DataLoader,
    device: torch.device | str,
    eps: float = 0.1,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    total = torch.zeros_like(model.svdd_center, device=device)
    count = 0
    for images, _ in loader:
        embeddings = model(images.to(device)).svdd_embedding
        total += embeddings.sum(dim=0)
        count += embeddings.shape[0]
    if was_training:
        model.train()
    if count == 0:
        raise MLXUserError("Cannot initialize the Deep SVDD center from an empty training loader.")
    center = total / count
    near_zero = center.abs() < eps
    signs = torch.where(center < 0, -torch.ones_like(center), torch.ones_like(center))
    center = torch.where(near_zero, signs * eps, center)
    model.svdd_center.copy_(center)
    return center


@torch.no_grad()
def collect_svdd_scores(
    model: JointDeepSVDDClassifier,
    loader: DataLoader,
    device: torch.device | str,
) -> torch.Tensor:
    model.eval()
    batches = []
    for images, _ in loader:
        output = model(images.to(device))
        batches.append(model.compute_svdd_score(output.svdd_embedding).cpu())
    if not batches:
        raise MLXUserError("Cannot calibrate Deep SVDD using an empty validation loader.")
    return torch.cat(batches)


def calibrate_svdd_threshold(
    model: JointDeepSVDDClassifier,
    scores: torch.Tensor,
    quantile: float,
) -> float:
    threshold = quantile_threshold(scores, quantile).to(model.svdd_threshold.device)
    model.svdd_threshold.copy_(threshold)
    return float(threshold.item())
