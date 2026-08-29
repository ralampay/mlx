from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError


class StandardClassifierAdapter(nn.Module):
    """Expose stable feature extraction without runtime forward hooks."""

    def __init__(self, model_name: str, model: nn.Module) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.feature_dim = self._resolve_feature_dim()

    def _resolve_feature_dim(self) -> int:
        if hasattr(self.model, "feature_dim"):
            return int(self.model.feature_dim)
        if self.model_name.startswith("resnet"):
            return int(self.model.fc.in_features)
        if self.model_name == "densenet121":
            return int(self.model.classifier.in_features)
        if self.model_name in {"mobilenet_v3_large", "efficientnet_b0"}:
            return int(next(layer for layer in self.model.classifier if isinstance(layer, nn.Linear)).in_features)
        if self.model_name.startswith("convnext_"):
            return int(self.model.classifier[2].in_features)
        raise MLXUserError(
            f"Model '{self.model_name}' has no Deep SVDD feature adapter. "
            "Custom classifiers must expose feature_dim, forward_features(), and classify_features()."
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(images)
        if self.model_name.startswith("resnet"):
            x = self.model.conv1(images)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            return torch.flatten(self.model.avgpool(x), 1)
        if self.model_name == "densenet121":
            x = F.relu(self.model.features(images), inplace=True)
            return torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        if self.model_name in {"mobilenet_v3_large", "efficientnet_b0"}:
            x = self.model.features(images)
            return torch.flatten(self.model.avgpool(x), 1)
        if self.model_name.startswith("convnext_"):
            x = self.model.features(images)
            x = self.model.avgpool(x)
            x = self.model.classifier[0](x)
            return self.model.classifier[1](x)
        raise MLXUserError(f"Model '{self.model_name}' has no Deep SVDD feature adapter.")

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "classify_features"):
            return self.model.classify_features(features)
        if self.model_name.startswith("resnet"):
            return self.model.fc(features)
        if self.model_name == "densenet121":
            return self.model.classifier(features)
        if self.model_name in {"mobilenet_v3_large", "efficientnet_b0"}:
            return self.model.classifier(features)
        if self.model_name.startswith("convnext_"):
            return self.model.classifier[2](features)
        raise MLXUserError(f"Model '{self.model_name}' has no Deep SVDD feature adapter.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(images))


def build_feature_adapter(model_name: str, model: nn.Module) -> StandardClassifierAdapter:
    return StandardClassifierAdapter(model_name, model)


class StandardImageFeatureBackbone(nn.Module):
    """Classification compatibility adapter implementing the neutral core contract."""

    def __init__(self, adapter: StandardClassifierAdapter) -> None:
        super().__init__()
        self.adapter = adapter
        self.feature_dim = adapter.feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.adapter.forward_features(images)


def build_image_feature_backbone(
    model_name: str,
    config: dict,
) -> StandardImageFeatureBackbone:
    # Local imports avoid a package initialization cycle and keep the registry authoritative.
    from mlx.modes.image_classification.models import (
        build_image_classification_model,
        model_family_for,
    )

    if model_family_for(model_name) != "standard":
        raise MLXUserError(
            f"Model '{model_name}' is a one-shot/Siamese model and cannot be used as an image feature backbone."
        )
    model = build_image_classification_model(
        model_name,
        {**config, "ood_method": "none"},
        num_classes=1,
    )
    adapter = build_feature_adapter(model_name, model)
    _remove_classification_head(model_name, model)
    return StandardImageFeatureBackbone(adapter)


def _remove_classification_head(model_name: str, model: nn.Module) -> None:
    """Remove only the final logits layer after the adapter captured its width."""

    if model_name.startswith("resnet") or model_name == "draxnet":
        model.fc = nn.Identity()
    elif model_name == "densenet121":
        model.classifier = nn.Identity()
    elif model_name in {"mobilenet_v3_large", "efficientnet_b0", "drax_mobilenet_v3_large"}:
        model.classifier = nn.Identity()
    elif model_name.startswith("convnext_"):
        model.classifier[2] = nn.Identity()
