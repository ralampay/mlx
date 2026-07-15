from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models.blocks import DraxBlock
from mlx.modes.image_classification.models.drax_mobilenet import (
    DraxMobileNetV3Large,
    build_drax_mobilenet_v3_large,
)
from mlx.modes.image_classification.models.draxnet import DraxResidualBlock, build_draxnet


class _ResidualDelta(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta


def test_average_fusion_preserves_existing_behavior_without_gate_parameters() -> None:
    block = DraxBlock(dim=4, efficient=False, fusion_mode="average")
    block.convnext = _ResidualDelta(2.0)
    block.attention = _ResidualDelta(4.0)
    inputs = torch.zeros(1, 4, 2, 2)

    output = block(inputs)

    assert torch.equal(output, torch.full_like(inputs, 3.0))
    assert block.fusion_gate is None
    assert not any(name.startswith("fusion_gate") for name, _ in block.named_parameters())


def test_sknet_fusion_produces_normalized_channel_weights_for_batch_size_one() -> None:
    block = DraxBlock(dim=4, efficient=False, fusion_mode="sknet")
    conv_delta = torch.randn(1, 4, 3, 3)
    attention_delta = torch.randn(1, 4, 3, 3)

    logits = block.fusion_gate(conv_delta + attention_delta)
    weights = logits.reshape(1, 2, 4, 1, 1).softmax(dim=1)
    fused = block._fuse_deltas(conv_delta, attention_delta)

    assert fused.shape == conv_delta.shape
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    assert torch.allclose(fused, weights[:, 0] * conv_delta + weights[:, 1] * attention_delta)


def test_sknet_fusion_gate_receives_gradients() -> None:
    block = DraxBlock(dim=4, efficient=False, fusion_mode="sknet")
    conv_delta = torch.randn(2, 4, 3, 3)
    attention_delta = torch.randn(2, 4, 3, 3)

    block._fuse_deltas(conv_delta, attention_delta).square().mean().backward()

    gate_gradients = [parameter.grad for parameter in block.fusion_gate.parameters()]
    assert all(gradient is not None for gradient in gate_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gate_gradients)


@pytest.mark.parametrize("fusion_mode", ["average", "sknet"])
def test_attention_disabled_remains_convolution_only(fusion_mode: str) -> None:
    block = DraxBlock(dim=4, use_attention=False, fusion_mode=fusion_mode)
    block.convnext = _ResidualDelta(2.0)
    inputs = torch.zeros(1, 4, 2, 2)

    assert torch.equal(block(inputs), torch.full_like(inputs, 2.0))
    assert block.fusion_gate is None


def test_direct_block_rejects_invalid_fusion_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported Drax fusion mode 'invalid'"):
        DraxBlock(fusion_mode="invalid")


def test_draxnet_builder_propagates_sknet_fusion() -> None:
    model = build_draxnet(
        num_classes=3,
        colored=True,
        pretrained=False,
        config={"draxnet_fusion_mode": "sknet"},
    )

    drax_blocks = [module for module in model.modules() if isinstance(module, DraxResidualBlock)]
    assert drax_blocks
    assert all(block.drax.fusion_mode == "sknet" for block in drax_blocks)
    assert all(block.drax.fusion_gate is not None for block in drax_blocks)


def test_generic_cli_fusion_config_propagates_to_draxnet() -> None:
    model = build_draxnet(
        num_classes=3,
        colored=True,
        pretrained=False,
        config={"drax_fusion_mode": "sknet"},
    )

    drax_blocks = [module for module in model.modules() if isinstance(module, DraxResidualBlock)]
    assert drax_blocks
    assert all(block.drax.fusion_mode == "sknet" for block in drax_blocks)


def test_draxnet_builder_reports_invalid_user_fusion_mode() -> None:
    with pytest.raises(MLXUserError, match="Unsupported Drax fusion mode 'invalid'"):
        build_draxnet(
            num_classes=3,
            colored=True,
            pretrained=False,
            config={"draxnet_fusion_mode": "invalid"},
        )


def test_drax_mobilenet_constructor_propagates_sknet_fusion() -> None:
    backbone = SimpleNamespace(
        features=nn.Identity(),
        avgpool=nn.AdaptiveAvgPool2d(1),
        classifier=nn.Sequential(nn.Linear(8, 8), nn.Hardswish(), nn.Dropout(), nn.Linear(8, 4)),
    )

    model = DraxMobileNetV3Large(
        backbone,
        num_classes=3,
        drax_blocks=2,
        adapter_dim=4,
        fusion_mode="sknet",
    )

    assert all(block.fusion_mode == "sknet" for block in model.drax_refiner)
    assert all(block.fusion_gate is not None for block in model.drax_refiner)


def test_generic_cli_fusion_config_propagates_to_drax_mobilenet() -> None:
    model = build_drax_mobilenet_v3_large(
        num_classes=3,
        colored=True,
        pretrained=False,
        config={"drax_fusion_mode": "sknet", "drax_mobilenet_adapter_dim": 32},
    )

    assert all(block.fusion_mode == "sknet" for block in model.drax_refiner)
    assert all(block.fusion_gate is not None for block in model.drax_refiner)


def test_drax_mobilenet_builder_reports_invalid_user_fusion_mode() -> None:
    with pytest.raises(MLXUserError, match="Unsupported Drax fusion mode 'invalid'"):
        build_drax_mobilenet_v3_large(
            num_classes=3,
            colored=True,
            pretrained=False,
            config={"drax_mobilenet_fusion_mode": "invalid"},
        )


def test_cli_accepts_adaptive_drax_fusion_mode() -> None:
    namespace = build_parser().parse_args(["--drax-fusion-mode", "sknet"])

    assert namespace.drax_fusion_mode == "sknet"
