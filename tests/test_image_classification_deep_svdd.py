from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification import inference, train
from mlx.modes.image_classification.models.adapters import build_feature_adapter
from mlx.modes.image_classification.models import build_image_classification_model
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier
from mlx.modes.image_classification.models.drax_mobilenet import DraxMobileNetV3Large
from mlx.modes.image_classification.models.draxnet import DraxNet
from mlx.modes.image_classification.ood.deep_svdd import (
    calibrate_svdd_threshold,
    compute_svdd_loss,
    initialize_svdd_center,
    validate_svdd_config,
)
from mlx.modes.image_classification.utils import (
    checkpoint_payload,
    load_checkpoint_bundle,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TinyClassifier(nn.Module):
    feature_dim = 4

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Linear(4, 4, bias=False)
        self.classifier = nn.Linear(4, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(images))


def _joint(svdd_dim: int = 2) -> JointDeepSVDDClassifier:
    return JointDeepSVDDClassifier(TinyClassifier(), 4, svdd_dim=svdd_dim, svdd_hidden_dim=3)


@pytest.mark.parametrize(
    ("model_name", "model_factory", "feature_dim"),
    [
        *[
            (
                name,
                lambda: SimpleNamespaceModule(
                    conv1=nn.Conv2d(3, 4, 1), bn1=nn.Identity(), relu=nn.ReLU(),
                    maxpool=nn.Identity(), layer1=nn.Identity(), layer2=nn.Identity(),
                    layer3=nn.Identity(), layer4=nn.Identity(), avgpool=nn.AdaptiveAvgPool2d(1),
                    fc=nn.Linear(4, 3),
                ),
                4,
            )
            for name in ("resnet18", "resnet50")
        ],
        (
            "densenet121",
            lambda: SimpleNamespaceModule(features=nn.Conv2d(3, 5, 1), classifier=nn.Linear(5, 3)),
            5,
        ),
        *[
            (
                name,
                lambda: SimpleNamespaceModule(
                    features=nn.Conv2d(3, 6, 1), avgpool=nn.AdaptiveAvgPool2d(1),
                    classifier=nn.Sequential(nn.Linear(6, 7), nn.ReLU(), nn.Dropout(), nn.Linear(7, 3)),
                ),
                6,
            )
            for name in ("mobilenet_v3_large", "efficientnet_b0")
        ],
        *[
            (
                name,
                lambda: SimpleNamespaceModule(
                    features=nn.Conv2d(3, 8, 1), avgpool=nn.AdaptiveAvgPool2d(1),
                    classifier=nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(8, 3)),
                ),
                8,
            )
            for name in ("convnext_tiny", "convnext_small", "convnext_base", "convnext_large")
        ],
    ],
)
def test_torchvision_family_adapters_return_vector_features_and_logits(
    model_name, model_factory, feature_dim
):
    adapter = build_feature_adapter(model_name, model_factory())
    images = torch.randn(2, 3, 8, 8)
    features = adapter.forward_features(images)
    assert features.shape == (2, feature_dim)
    assert adapter.classify_features(features).shape == (2, 3)
    assert adapter(images).shape == (2, 3)
    assert adapter.feature_dim == feature_dim


class SimpleNamespaceModule(nn.Module):
    def __init__(self, **modules) -> None:
        super().__init__()
        for name, module in modules.items():
            setattr(self, name, module)


def test_joint_wrapper_shapes_center_and_gradients():
    model = _joint(svdd_dim=5)
    inputs = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])
    output = model(inputs)
    assert output.logits.shape == (3, 3)
    assert output.svdd_embedding.shape == (3, 5)
    assert "svdd_center" in model.state_dict()
    loss = nn.functional.cross_entropy(output.logits, labels) + 0.05 * compute_svdd_loss(
        model, output.svdd_embedding
    )
    loss.backward()
    assert model.classifier.classifier.weight.grad is not None
    assert model.svdd_head[0].weight.grad is not None
    assert model.svdd_center.grad is None
    assert "svdd_center" not in dict(model.named_parameters())


def test_draxnet_feature_interface_preserves_ordinary_forward():
    model = DraxNet(in_channels=3, num_classes=3, stage_block_types=("basic",) * 4)
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = model.forward_features(images)
        logits = model.classify_features(features)
        ordinary_logits = model(images)
    assert features.shape == (2, model.feature_dim)
    assert logits.shape == (2, 3)
    assert torch.equal(logits, ordinary_logits)


def test_ordinary_torchvision_model_still_returns_logits():
    model = build_image_classification_model(
        "resnet18",
        {"colored": True, "pretrained": False, "ood_method": "none"},
        num_classes=3,
    ).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 3)


def test_drax_mobilenet_feature_interface_preserves_ordinary_forward():
    backbone = SimpleNamespaceModule(
        features=nn.Conv2d(3, 8, 1),
        avgpool=nn.AdaptiveAvgPool2d(1),
        classifier=nn.Sequential(nn.Linear(8, 10), nn.Hardswish(), nn.Dropout(), nn.Linear(10, 3)),
    )
    model = DraxMobileNetV3Large(backbone, num_classes=3, adapter_dim=8)
    model.eval()
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        features = model.forward_features(images)
        logits = model.classify_features(features)
        ordinary_logits = model(images)
    assert features.shape == (2, model.feature_dim)
    assert logits.shape == (2, 3)
    assert torch.equal(logits, ordinary_logits)


def test_zero_svdd_weight_matches_classification_only_loss():
    model = _joint()
    output = model(torch.randn(2, 4))
    labels = torch.tensor([0, 1])
    classification_loss = nn.functional.cross_entropy(output.logits, labels)
    total = classification_loss + 0.0 * compute_svdd_loss(model, output.svdd_embedding)
    assert torch.equal(total, classification_loss)


def test_center_initialization_uses_all_batches_and_copies_buffer():
    model = _joint()
    with torch.no_grad():
        model.classifier.features.weight.copy_(torch.eye(4))
        model.svdd_head[0].weight.fill_(0.5)
        model.svdd_head[2].weight.fill_(0.5)
    images = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1
    loader = DataLoader(TensorDataset(images, torch.zeros(4, dtype=torch.long)), batch_size=2)
    expected = torch.cat([model(batch).svdd_embedding for batch, _ in loader]).mean(dim=0)
    center = initialize_svdd_center(model, loader, "cpu", eps=1e-6)
    assert torch.allclose(center, expected)
    assert torch.allclose(model.svdd_center, expected)


def test_center_initialization_rejects_empty_loader():
    loader = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    with pytest.raises(MLXUserError, match="empty training loader"):
        initialize_svdd_center(_joint(), loader, "cpu")


def test_threshold_calibration_quantile_and_checkpoint_metadata():
    model = _joint()
    threshold = calibrate_svdd_threshold(model, torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.75)
    assert threshold == pytest.approx(torch.quantile(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.75).item())
    payload = checkpoint_payload(
        model,
        model_name="tiny",
        family="standard",
        config={"input_size": (4, 4), "ood_method": "deep-svdd", "svdd_quantile": 0.75},
        classes=["a", "b", "c"],
    )
    assert payload["ood"]["threshold"] == pytest.approx(threshold)
    assert torch.equal(payload["ood"]["center"], model.svdd_center.cpu())
    restored = _joint()
    restored.load_state_dict(payload["state_dict"])
    assert restored.svdd_threshold.item() == pytest.approx(threshold)


def test_ordinary_checkpoint_format_omits_svdd_metadata_and_defaults():
    payload = checkpoint_payload(
        TinyClassifier(),
        model_name="tiny",
        family="standard",
        config={
            "input_size": (4, 4), "ood_method": "none", "svdd_dim": 128,
            "svdd_hidden_dim": 256, "svdd_weight": 0.05, "svdd_quantile": 0.95,
            "svdd_warmup_epochs": 0,
        },
        classes=["a", "b", "c"],
    )
    assert "ood" not in payload
    assert "ood_method" not in payload["model_config"]
    assert "svdd_dim" not in payload["model_config"]


def test_old_style_checkpoint_without_ood_loads_normally(tmp_path):
    config = {
        "model": "resnet18", "colored": True, "pretrained": False,
        "input_size": (32, 32), "ood_method": "none", "device": "cpu",
    }
    model = build_image_classification_model("resnet18", config, num_classes=3)
    payload = checkpoint_payload(
        model, model_name="resnet18", family="standard", config=config,
        classes=["a", "b", "c"],
    )
    payload.pop("ood", None)
    path = tmp_path / "old.pth"
    torch.save(payload, path)
    restored, metadata = load_checkpoint_bundle({**config, "model_path": str(path)})
    assert not isinstance(restored, JointDeepSVDDClassifier)
    assert metadata["ood"]["method"] == "none"


def test_resumable_checkpoint_restores_exact_center(tmp_path):
    model = _joint()
    model.svdd_center.copy_(torch.tensor([1.25, -2.5]))
    optimizer = torch.optim.Adam(model.parameters())
    config = {
        "colored": True,
        "input_size": (4, 4),
        "device": "cpu",
        "drax_fusion_mode": "average",
        "ood_method": "deep-svdd",
        "svdd_dim": 2,
        "svdd_hidden_dim": 3,
    }
    path = tmp_path / "tiny.last.pth"
    save_training_checkpoint(
        path, model, optimizer, model_name="tiny", family="standard", config=config,
        completed_epoch=0, best_val_loss=float("inf"), history=[], classes=["a", "b", "c"],
    )
    restored = _joint()
    load_training_checkpoint(
        path, restored, torch.optim.Adam(restored.parameters()), model_name="tiny",
        family="standard", config=config, classes=["a", "b", "c"],
    )
    assert torch.equal(restored.svdd_center, model.svdd_center)


def test_svdd_resume_rejects_ordinary_checkpoint(tmp_path):
    model = TinyClassifier()
    ordinary_config = {
        "colored": True, "input_size": (4, 4), "device": "cpu",
        "drax_fusion_mode": "average", "ood_method": "none",
    }
    path = tmp_path / "ordinary.last.pth"
    save_training_checkpoint(
        path, model, torch.optim.Adam(model.parameters()), model_name="tiny",
        family="standard", config=ordinary_config, completed_epoch=0,
        best_val_loss=float("inf"), history=[], classes=["a", "b", "c"],
    )
    joint = _joint()
    svdd_config = {
        **ordinary_config, "ood_method": "deep-svdd", "svdd_dim": 2, "svdd_hidden_dim": 3,
    }
    with pytest.raises(MLXUserError, match="OOD method"):
        load_training_checkpoint(
            path, joint, torch.optim.Adam(joint.parameters()), model_name="tiny",
            family="standard", config=svdd_config, classes=["a", "b", "c"],
        )


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1])
def test_invalid_quantiles_are_rejected(quantile):
    with pytest.raises(MLXUserError, match="strictly between"):
        calibrate_svdd_threshold(_joint(), torch.ones(2), quantile)


def test_svdd_configuration_validation():
    for config in (
        {"ood_method": "deep-svdd", "svdd_weight": -1},
        {"ood_method": "deep-svdd", "svdd_dim": 0},
        {"ood_method": "deep-svdd", "svdd_hidden_dim": 0},
        {"ood_method": "deep-svdd", "svdd_quantile": 1},
        {"ood_method": "deep-svdd", "svdd_warmup_epochs": -1},
    ):
        with pytest.raises(MLXUserError):
            validate_svdd_config(config)
    validate_svdd_config({"ood_method": "none", "svdd_dim": 0})


def test_inference_accepts_and_rejects_without_exposing_rejected_label(monkeypatch):
    model = _joint()
    with torch.no_grad():
        model.classifier.features.weight.copy_(torch.eye(4))
        model.svdd_head[0].weight.fill_(1.0)
        model.svdd_head[2].weight.fill_(1.0)
        model.svdd_center.zero_()
        model.svdd_threshold.fill_(0.5)
    metadata = {
        "family": "standard", "classes": ["a", "b", "c"],
        "input_size": (1, 4), "colored": True,
    }
    monkeypatch.setattr(inference, "load_checkpoint_bundle", lambda config: (model, metadata))
    monkeypatch.setattr(inference, "display_classification_predictions", lambda result: None)

    monkeypatch.setattr(inference, "load_image_tensor", lambda *args, **kwargs: torch.zeros(4))
    accepted = inference.infer_image_classification({"device": "cpu", "input_img": "accepted.png"})
    assert accepted["accepted"] is True
    assert accepted["predicted_label"] in metadata["classes"]
    assert accepted["confidence"] is not None
    assert accepted["rejection_reason"] is None

    monkeypatch.setattr(inference, "load_image_tensor", lambda *args, **kwargs: torch.full((4,), 100.0))
    rejected = inference.infer_image_classification({"device": "cpu", "input_img": "rejected.png"})
    assert rejected["accepted"] is False
    assert rejected["predicted_label"] is None
    assert rejected["confidence"] is None
    assert rejected["top_predictions"] == []
    assert rejected["ood_score"] > rejected["ood_threshold"]
    assert rejected["rejection_reason"] == "out_of_distribution"


def test_non_svdd_inference_output_remains_unchanged(monkeypatch):
    model = TinyClassifier()
    metadata = {
        "family": "standard", "classes": ["a", "b", "c"],
        "input_size": (1, 4), "colored": True,
    }
    monkeypatch.setattr(inference, "load_checkpoint_bundle", lambda config: (model, metadata))
    monkeypatch.setattr(inference, "display_classification_predictions", lambda result: None)
    monkeypatch.setattr(inference, "load_image_tensor", lambda *args, **kwargs: torch.zeros(4))
    result = inference.infer_image_classification({"device": "cpu", "input_img": "image.png"})
    assert set(result) == {"input_image", "predicted_label", "top_predictions"}


def test_inference_requires_calibrated_threshold(monkeypatch):
    model = _joint()
    metadata = {"family": "standard", "classes": ["a", "b", "c"], "input_size": (1, 4), "colored": True}
    monkeypatch.setattr(inference, "load_checkpoint_bundle", lambda config: (model, metadata))
    monkeypatch.setattr(inference, "load_image_tensor", lambda *args, **kwargs: torch.zeros(4))
    with pytest.raises(MLXUserError, match="no calibrated rejection threshold"):
        inference.infer_image_classification({"device": "cpu", "input_img": str(Path("image.png"))})


def test_joint_training_calibrates_final_checkpoint_and_extends_history(monkeypatch, tmp_path):
    images = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    dataset = TensorDataset(images, labels)
    monkeypatch.setattr(
        train,
        "load_standard_classification_datasets",
        lambda *args, **kwargs: (dataset, dataset, ["a", "b", "c"]),
    )
    monkeypatch.setattr(train, "build_image_classification_model", lambda *args, **kwargs: _joint())
    config = {
        "apply_transformations": False,
        "batch_size": 2,
        "colored": True,
        "dataset_path": str(tmp_path / "dataset"),
        "device": "cpu",
        "drax_fusion_mode": "average",
        "epochs": 1,
        "input_size": (1, 4),
        "lr": 0.001,
        "model_path": None,
        "ood_method": "deep-svdd",
        "output_path": str(tmp_path / "artifacts"),
        "pretrained": False,
        "refresh_per_second": 2,
        "svdd_dim": 2,
        "svdd_hidden_dim": 3,
        "svdd_quantile": 0.75,
        "svdd_warmup_epochs": 0,
        "svdd_weight": 0.05,
        "use_best": True,
        "verbose": False,
    }
    train._train_standard("tiny", config)
    final = torch.load(tmp_path / "artifacts" / "tiny.pth", map_location="cpu", weights_only=True)
    resumable = torch.load(tmp_path / "artifacts" / "tiny.last.pth", map_location="cpu", weights_only=True)
    assert final["ood"]["threshold"] is not None
    assert torch.equal(final["ood"]["center"], final["state_dict"]["svdd_center"])
    assert torch.equal(resumable["ood"]["center"], resumable["state_dict"]["svdd_center"])
    header = (tmp_path / "artifacts" / "training.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "train_classification_loss" in header
    assert "train_svdd_loss" in header
    assert "val_classification_loss" in header
    assert "val_svdd_loss" in header
