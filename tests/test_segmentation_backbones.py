from __future__ import annotations

import gc

import pytest
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models.blocks import DraxBlock
from mlx.modes.segmentation import list_models as segmentation_listing
from mlx.modes.segmentation.models import (
    MODEL_NAMES,
    BackboneUNet,
    build_segmentation_model,
    supported_model_names,
)
from mlx.modes.segmentation.models.backbones import BACKBONE_SPECS


EXPECTED_BACKBONE_MODELS = {
    "unet-resnet18",
    "unet-resnet50",
    "unet-densenet121",
    "unet-mobilenet_v3_large",
    "unet-efficientnet_b0",
    "unet-convnext_tiny",
    "unet-convnext_small",
    "unet-convnext_base",
    "unet-convnext_large",
    "unet-draxnet-average",
    "unet-draxnet-sknet",
    "unet-drax_mobilenet_v3_large-average",
    "unet-drax_mobilenet_v3_large-sknet",
}


def test_segmentation_registry_contains_all_non_siamese_backbones() -> None:
    assert set(BACKBONE_SPECS) == EXPECTED_BACKBONE_MODELS
    assert MODEL_NAMES == {"unet", *EXPECTED_BACKBONE_MODELS}
    assert supported_model_names() == sorted(MODEL_NAMES)
    assert not any("siamese" in model_name for model_name in MODEL_NAMES)


@pytest.mark.parametrize(
    "model_name",
    [
        "unet-resnet18",
        "unet-resnet50",
        "unet-densenet121",
        "unet-mobilenet_v3_large",
        "unet-efficientnet_b0",
        "unet-convnext_tiny",
        "unet-draxnet-average",
        "unet-draxnet-sknet",
        "unet-drax_mobilenet_v3_large-average",
        "unet-drax_mobilenet_v3_large-sknet",
    ],
)
def test_backbone_unets_restore_odd_input_size(model_name: str) -> None:
    model = build_segmentation_model(
        model_name,
        {"colored": True, "pretrained": False},
        num_classes=3,
    ).eval()

    with torch.inference_mode():
        output = model(torch.randn(1, 3, 33, 35))

    assert isinstance(model, BackboneUNet)
    assert output.shape == (1, 3, 33, 35)
    del model, output
    gc.collect()


@pytest.mark.parametrize(
    "model_name",
    [
        "unet-convnext_small",
        "unet-convnext_base",
        "unet-convnext_large",
    ],
)
def test_all_convnext_width_variants_build_with_declared_channels(model_name: str) -> None:
    model = build_segmentation_model(
        model_name,
        {"colored": True, "pretrained": False},
        num_classes=2,
    )

    assert tuple(model.encoder.output_channels) == BACKBONE_SPECS[model_name].output_channels
    del model
    gc.collect()


def test_backbone_unet_supports_grayscale_input() -> None:
    model = build_segmentation_model(
        "unet-resnet18",
        {"colored": False, "pretrained": False},
        num_classes=2,
    ).eval()

    with torch.inference_mode():
        output = model(torch.randn(1, 1, 33, 35))

    assert model.encoder.stem[0].in_channels == 1
    assert output.shape == (1, 2, 33, 35)


@pytest.mark.parametrize(
    ("model_name", "fusion_mode", "has_gate"),
    [
        ("unet-draxnet-average", "average", False),
        ("unet-draxnet-sknet", "sknet", True),
        ("unet-drax_mobilenet_v3_large-average", "average", False),
        ("unet-drax_mobilenet_v3_large-sknet", "sknet", True),
    ],
)
def test_explicit_drax_variants_control_fusion(
    model_name: str,
    fusion_mode: str,
    has_gate: bool,
) -> None:
    model = build_segmentation_model(
        model_name,
        {
            "colored": True,
            "pretrained": False,
            "drax_fusion_mode": "sknet" if fusion_mode == "average" else "average",
        },
        num_classes=2,
    )
    drax_blocks = [module for module in model.modules() if isinstance(module, DraxBlock)]

    assert drax_blocks
    assert all(block.fusion_mode == fusion_mode for block in drax_blocks)
    assert all((block.fusion_gate is not None) is has_gate for block in drax_blocks)


def test_sknet_variants_have_more_parameters_than_average_variants() -> None:
    average = build_segmentation_model(
        "unet-draxnet-average",
        {"colored": True, "pretrained": False},
        num_classes=2,
    )
    sknet = build_segmentation_model(
        "unet-draxnet-sknet",
        {"colored": True, "pretrained": False},
        num_classes=2,
    )

    assert sum(parameter.numel() for parameter in sknet.parameters()) > sum(
        parameter.numel() for parameter in average.parameters()
    )


def test_segmentation_listing_disables_pretrained_downloads(monkeypatch) -> None:
    configs = []

    monkeypatch.setattr(
        segmentation_listing,
        "supported_model_names",
        lambda: ["unet-resnet18"],
    )

    def fake_build(model_name, config, *, num_classes):
        configs.append(config)
        return torch.nn.Linear(2, num_classes)

    monkeypatch.setattr(segmentation_listing, "build_segmentation_model", fake_build)

    segmentation_listing.ListSegmentationModels(
        {"pretrained": True, "num_classes": 2}
    ).execute()

    assert configs[0]["pretrained"] is False


def test_pretrained_draxnet_backbone_failure_is_user_facing() -> None:
    with pytest.raises(
        MLXUserError,
        match="Pretrained DraxNet weights currently require all stages",
    ):
        build_segmentation_model(
            "unet-draxnet-average",
            {"colored": True, "pretrained": True},
            num_classes=2,
        )
