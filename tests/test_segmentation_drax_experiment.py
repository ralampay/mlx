from __future__ import annotations

import pytest
import torch

from mlx.modes.segmentation.losses import CrossEntropyDiceLoss, build_loss
from mlx.modes.segmentation.models import build_segmentation_model


def test_sparse_foreground_loss_is_finite_and_propagates_gradients() -> None:
    logits = torch.zeros(2, 2, 8, 8, requires_grad=True)
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    targets[:, 3, 4] = 1
    loss = build_loss({"loss": "cross-entropy-dice"})(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[:, 1, 3, 4].abs().sum() > 0


def test_sparse_foreground_loss_rejects_nonbinary_logits() -> None:
    with pytest.raises(ValueError, match="two output classes"):
        CrossEntropyDiceLoss()(torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long))


def test_foreground_weight_increases_penalty_for_missed_defect() -> None:
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 0] = 3
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[:, 2, 2] = 1
    plain = CrossEntropyDiceLoss(foreground_weight=1)(logits, targets)
    weighted = CrossEntropyDiceLoss(foreground_weight=20)(logits, targets)
    assert weighted > plain


@pytest.mark.parametrize("name", [
    "unet-mobilenet_v3_large-skip-conv",
    "unet-mobilenet_v3_large-skip-drax",
    "unet-mobilenet_v3_large-skip-drax-balanced",
])
def test_new_variants_round_trip_state_dict(name: str) -> None:
    config = {"colored": True, "pretrained": False}
    original = build_segmentation_model(name, config, num_classes=2)
    restored = build_segmentation_model(name, config, num_classes=2)
    restored.load_state_dict(original.state_dict())
    assert set(restored.state_dict()) == set(original.state_dict())
