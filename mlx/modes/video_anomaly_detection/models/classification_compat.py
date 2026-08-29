from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from mlx.modes.image_classification.models.adapters import build_image_feature_backbone
from mlx.modes.image_classification.models.blocks import ConvNeXtBlock, DraxBlock


ClassificationModuleKind = Literal["convnext", "drax"]


class InflatableClassificationBackbone(nn.Module):
    """Stable video-facing wrapper around a standard classifier implementation."""

    def __init__(self, model_name: str, model: nn.Module, feature_dim: int, provenance: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.feature_dim = int(feature_dim)
        self.pretrained_provenance = provenance

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(inputs)
        if self.model_name.startswith("resnet"):
            x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(inputs))))
            x = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
            return torch.flatten(self.model.avgpool(x), 1)
        if self.model_name == "densenet121":
            x = F.relu(self.model.features(inputs), inplace=True)
            output_size = (1, 1, 1) if x.ndim == 5 else (1, 1)
            pool = F.adaptive_avg_pool3d if x.ndim == 5 else F.adaptive_avg_pool2d
            return torch.flatten(pool(x, output_size), 1)
        if self.model_name in {"mobilenet_v3_large", "efficientnet_b0"}:
            return torch.flatten(self.model.avgpool(self.model.features(inputs)), 1)
        if self.model_name.startswith("convnext_"):
            x = self.model.avgpool(self.model.features(inputs))
            x = self.model.classifier[0](x)
            return self.model.classifier[1](x)
        raise RuntimeError(f"No feature path is available for '{self.model_name}'.")


def build_frame_feature_backbone(model_name: str, config: dict[str, Any]):
    return build_image_feature_backbone(model_name, config)


def build_inflatable_backbone(
    model_name: str,
    config: dict[str, Any],
) -> InflatableClassificationBackbone:
    pretrained = bool(config.get("pretrained", False))
    source_config = {**config, "colored": True}
    if model_name == "draxnet" and pretrained:
        wrapper = build_image_feature_backbone(
            model_name, {**source_config, "pretrained": False}
        )
        _load_partial_resnet18_weights(wrapper.adapter.model)
    else:
        wrapper = build_image_feature_backbone(model_name, source_config)
    provenance = "none"
    if pretrained:
        provenance = "inflated_partial" if model_name.startswith("drax") else "inflated_full"
    return InflatableClassificationBackbone(
        model_name,
        wrapper.adapter.model,
        int(wrapper.feature_dim),
        provenance,
    )


def classification_module_kind(module: nn.Module) -> ClassificationModuleKind | None:
    if isinstance(module, DraxBlock):
        return "drax"
    if isinstance(module, ConvNeXtBlock):
        return "convnext"
    return None


def standard_backbone_names() -> tuple[str, ...]:
    from mlx.modes.image_classification.models import standard_model_names

    return tuple(standard_model_names())


def is_standard_backbone(model_name: str) -> bool:
    from mlx.modes.image_classification.models import model_family_for

    return model_family_for(model_name) == "standard"


def _load_partial_resnet18_weights(model: nn.Module) -> None:
    from mlx.core.exceptions import MLXUserError

    try:
        from torchvision import models as torchvision_models

        reference = torchvision_models.resnet18(
            weights=torchvision_models.ResNet18_Weights.DEFAULT
        )
    except (ImportError, OSError, RuntimeError) as exc:
        raise MLXUserError(
            "Pretrained DraxNet3D initialization requires torchvision ResNet-18 weights."
        ) from exc
    target_state = model.state_dict()
    compatible = {
        name: value
        for name, value in reference.state_dict().items()
        if name in target_state and target_state[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)


__all__ = [
    "InflatableClassificationBackbone",
    "build_frame_feature_backbone",
    "build_inflatable_backbone",
    "classification_module_kind",
    "is_standard_backbone",
    "standard_backbone_names",
]
