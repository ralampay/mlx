from pathlib import Path

import pytest
import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification import inference
from mlx.modes.image_classification.cam import _default_target_layer_path
from mlx.modes.image_classification.models import (
    SIAMESE_BACKBONE_MODELS,
    SiameseBackbone,
    build_image_classification_model,
    model_family_for,
)


class TinyBackbone(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.features(x)).flatten(1))


@pytest.mark.parametrize("model_name", sorted(SIAMESE_BACKBONE_MODELS))
def test_registered_siamese_backbones_build_with_shared_contract(monkeypatch, model_name):
    calls = []

    def fake_build(backbone_name, **kwargs):
        calls.append((backbone_name, kwargs))
        return TinyBackbone(kwargs["num_classes"])

    monkeypatch.setattr("mlx.modes.image_classification.models.build_standard_model", fake_build)
    model = build_image_classification_model(
        model_name,
        {"colored": True, "embedding_size": 7, "pretrained": True},
    )

    first = torch.randn(2, 3, 8, 8)
    second = torch.randn(2, 3, 8, 8)
    assert model_family_for(model_name) == "one-shot"
    assert model.forward_once(first).shape == (2, 7)
    output = model(first, second)
    assert output.shape == (2, 1)
    assert torch.all((0 <= output) & (output <= 1))
    assert calls == [
        (
            SIAMESE_BACKBONE_MODELS[model_name],
            {
                "num_classes": 7,
                "colored": True,
                "pretrained": True,
                "config": {"colored": True, "embedding_size": 7, "pretrained": True},
            },
        )
    ]


def test_siamese_backbone_uses_one_module_for_both_branches():
    backbone = TinyBackbone(5)
    model = SiameseBackbone(backbone, embedding_size=5)

    assert model.embedding is backbone
    assert model(torch.ones(1, 3, 4, 4), torch.ones(1, 3, 4, 4)).shape == (1, 1)


def test_invalid_embedding_size_is_user_facing():
    with pytest.raises(MLXUserError, match="embedding-size must be at least 1"):
        build_image_classification_model("siamese-resnet18", {"embedding_size": 0})


def test_reference_inference_ranks_by_comparator_probability(monkeypatch, tmp_path):
    query_path = tmp_path / "query.png"
    low_path = tmp_path / "low.png"
    high_path = tmp_path / "high.png"
    paths = [low_path, high_path]
    values = {query_path: 0.0, low_path: 0.2, high_path: 0.9}

    monkeypatch.setattr(inference, "iter_dataset_images", lambda _: iter(paths))
    monkeypatch.setattr(
        inference,
        "load_image_tensor",
        lambda path, **_: torch.full((3, 2, 2), values[path]),
    )
    monkeypatch.setattr(inference, "display_similarity_matches", lambda result: None)

    class Comparator(nn.Module):
        def forward(self, query, reference):
            return reference.mean(dim=(1, 2, 3), keepdim=True)

    result = inference.RankOneShotReferences(
        model=Comparator(),
        metadata={"input_size": (2, 2), "colored": True},
        input_image=query_path,
        dataset_path=tmp_path,
        device="cpu",
    ).execute()

    assert result["best_match_path"] == high_path
    assert result["similarity_score"] == pytest.approx(0.9)
    assert [match[1] for match in result["top_matches"]] == [high_path, low_path]
    assert "distance" not in result


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("siamese-resnet18", "embedding.layer4.-1"),
        ("siamese-densenet121", "embedding.features.-1"),
        ("siamese-drax_mobilenet_v3_large", "embedding.adapter_up"),
    ],
)
def test_siamese_cam_defaults_are_nested(model_name, expected):
    model = nn.Module()
    model.embedding = nn.Module()
    if "resnet" in model_name:
        model.embedding.layer4 = nn.Sequential(nn.Identity())
    elif model_name == "siamese-drax_mobilenet_v3_large":
        model.embedding.adapter_up = nn.Conv2d(1, 1, 1)
    else:
        model.embedding.features = nn.Sequential(nn.Identity())

    assert _default_target_layer_path(model, model_name) == expected
