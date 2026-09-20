"""Compatibility checks for discovery and the merged mode inventories."""

import pytest

from mlx.cli import build_parser
from mlx.cli_routing import resolve_mode_descriptor, resolve_mode_runner


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("text-embedding", "text_embedding"),
        ("text_embedding", "text_embedding"),
        ("nlp", "text_embedding"),
        ("saliency-mapping", "saliency_mapping"),
        ("autoencoder", "autoencoder"),
    ],
)
def test_merged_mode_aliases(mode, expected):
    assert resolve_mode_descriptor(mode).name == expected


@pytest.mark.parametrize("action", ["embed", "benchmark"])
def test_text_embedding_invocations_parse(action):
    arguments = [
        "--mode", "text-embedding", "--action", action,
        "--input", "./dataset", "--output", "./output",
    ]
    if action == "embed":
        arguments.extend(["--model", "model.gguf"])
    config = build_parser().parse_args(arguments)
    assert config.input_path == "./dataset"
    assert config.action == action


def test_saliency_flags_and_custom_cam_references_coexist():
    config = build_parser().parse_args([
        "--bce-weight", "2", "--ssim-weight", "0.5", "--iou-weight", "0",
        "--curve-bins", "128", "--cam-method", "custom.cam:TinyCAM",
    ])
    assert (config.bce_weight, config.ssim_weight, config.iou_weight) == (2, 0.5, 0)
    assert config.curve_bins == 128
    assert config.cam_method == "custom.cam:TinyCAM"


@pytest.mark.parametrize("mode", [
    "image-classification", "segmentation", "video-anomaly-detection",
    "image-recognition-oc", "object-detection",
])
def test_names_only_does_not_construct_torch_models(mode, monkeypatch):
    from torch import nn

    def reject_construction(*args, **kwargs):
        raise AssertionError("Metadata discovery must not construct a model")

    runner = resolve_mode_runner(mode)
    monkeypatch.setattr(nn.Module, "__init__", reject_construction)
    results = runner({
        "action": "ls-models", "names_only": True, "output_format": "json",
        "provider": "ultralytics",
    })
    assert results
    assert [item.name for item in results] == sorted(item.name for item in results)


def test_native_3d_backbone_registration_does_not_require_a_classifier():
    from mlx.modes.video_anomaly_detection.models.backbone3d import (
        Backbone3DRegistry,
        build_spatiotemporal_backbone_3d,
    )

    calls = []
    sentinel = object()

    def build_native(name, config):
        calls.append((name, config))
        return sentinel

    empty = Backbone3DRegistry({})
    registry = empty.register("tiny-native", build_native)
    config = {"pretrained": False}
    assert build_spatiotemporal_backbone_3d(
        "tiny-native", config, registry=registry,
    ) is sentinel
    assert calls == [("tiny-native", config)]
    assert not empty.entries
