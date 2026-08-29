from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models.adapters import build_image_feature_backbone
from mlx.modes.image_classification.models.blocks import ConvNeXtBlock, DraxBlock
from mlx.modes.video_anomaly_detection.models.blocks3d import (
    LayerNorm3D,
    inflate_convnext_block,
    inflate_drax_block,
)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return value, value


def _optional_pair(value: int | tuple[int, int] | None, fallback) -> tuple[int, int]:
    return _pair(fallback if value is None else value)


def inflate_conv2d(source: nn.Conv2d, temporal_kernel_size: int) -> nn.Conv3d:
    """Inflate a 2D convolution while preserving its response to repeated frames."""

    spatial_kernel = _pair(source.kernel_size)
    temporal_kernel = temporal_kernel_size if spatial_kernel != (1, 1) else 1
    spatial_stride = _pair(source.stride)
    spatial_padding = _pair(source.padding)
    spatial_dilation = _pair(source.dilation)
    target = nn.Conv3d(
        source.in_channels,
        source.out_channels,
        kernel_size=(temporal_kernel, *spatial_kernel),
        stride=(1, *spatial_stride),
        padding=(temporal_kernel // 2, *spatial_padding),
        dilation=(1, *spatial_dilation),
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    )
    with torch.no_grad():
        target.weight.copy_(
            source.weight.unsqueeze(2).repeat(1, 1, temporal_kernel, 1, 1)
            / temporal_kernel
        )
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)
    return target


def _inflate_batch_norm(source: nn.BatchNorm2d) -> nn.BatchNorm3d:
    target = nn.BatchNorm3d(
        source.num_features,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    )
    target.load_state_dict(source.state_dict())
    return target


def _inflate_max_pool(source: nn.MaxPool2d) -> nn.MaxPool3d:
    kernel = _pair(source.kernel_size)
    stride = _optional_pair(source.stride, source.kernel_size)
    padding = _pair(source.padding)
    dilation = _pair(source.dilation)
    return nn.MaxPool3d(
        kernel_size=(1, *kernel),
        stride=(1, *stride),
        padding=(0, *padding),
        dilation=(1, *dilation),
        return_indices=source.return_indices,
        ceil_mode=source.ceil_mode,
    )


def _inflate_avg_pool(source: nn.AvgPool2d) -> nn.AvgPool3d:
    kernel = _pair(source.kernel_size)
    stride = _optional_pair(source.stride, source.kernel_size)
    padding = _pair(source.padding)
    return nn.AvgPool3d(
        kernel_size=(1, *kernel),
        stride=(1, *stride),
        padding=(0, *padding),
        ceil_mode=source.ceil_mode,
        count_include_pad=source.count_include_pad,
        divisor_override=source.divisor_override,
    )


def _inflate_adaptive_avg_pool(source: nn.AdaptiveAvgPool2d) -> nn.AdaptiveAvgPool3d:
    output = _pair(source.output_size)
    return nn.AdaptiveAvgPool3d((1, *output))


class TorchvisionConvNeXtBlock3D(nn.Module):
    """3D equivalent of torchvision's channel-last ConvNeXt block."""

    def __init__(self, source: nn.Module, temporal_kernel_size: int) -> None:
        super().__init__()
        self.depthwise = inflate_conv2d(source.block[0], temporal_kernel_size)
        self.norm = copy.deepcopy(source.block[2])
        self.linear1 = copy.deepcopy(source.block[3])
        self.activation = copy.deepcopy(source.block[4])
        self.linear2 = copy.deepcopy(source.block[5])
        self.layer_scale = nn.Parameter(source.layer_scale.detach().clone().unsqueeze(2))
        self.stochastic_depth = copy.deepcopy(source.stochastic_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 4, 1)
        x = self.linear2(self.activation(self.linear1(self.norm(x))))
        x = x.permute(0, 4, 1, 2, 3)
        return residual + self.stochastic_depth(self.layer_scale * x)


def _is_torchvision_convnext_block(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "CNBlock"
        and hasattr(module, "block")
        and hasattr(module, "layer_scale")
        and hasattr(module, "stochastic_depth")
    )


def inflate_module_2d_to_3d(module: nn.Module, temporal_kernel_size: int) -> nn.Module:
    """Recursively replace spatial operators with temporal-preserving 3D equivalents."""

    if isinstance(module, DraxBlock):
        return inflate_drax_block(module, temporal_kernel_size)
    if isinstance(module, ConvNeXtBlock):
        return inflate_convnext_block(module, temporal_kernel_size)
    if _is_torchvision_convnext_block(module):
        return TorchvisionConvNeXtBlock3D(module, temporal_kernel_size)
    if module.__class__.__name__ == "LayerNorm2d" and isinstance(module, nn.LayerNorm):
        target = LayerNorm3D(int(module.normalized_shape[0]), eps=module.eps)
        target.norm.load_state_dict(module.state_dict())
        return target
    if isinstance(module, nn.Conv2d):
        return inflate_conv2d(module, temporal_kernel_size)
    if isinstance(module, nn.BatchNorm2d):
        return _inflate_batch_norm(module)
    if isinstance(module, nn.MaxPool2d):
        return _inflate_max_pool(module)
    if isinstance(module, nn.AvgPool2d):
        return _inflate_avg_pool(module)
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return _inflate_adaptive_avg_pool(module)

    for name, child in tuple(module.named_children()):
        setattr(module, name, inflate_module_2d_to_3d(child, temporal_kernel_size))
    return module


class InflatedImageBackbone3D(nn.Module):
    """Clip-native feature backbone constructed from an image-recognition family."""

    supported_model_names: tuple[str, ...] = ()

    def __init__(self, model_name: str, config: dict[str, Any]) -> None:
        super().__init__()
        if model_name not in self.supported_model_names:
            supported = ", ".join(self.supported_model_names)
            raise MLXUserError(
                f"{type(self).__name__} cannot build '{model_name}'. Supported aliases: {supported}."
            )
        temporal_kernel_size = int(config.get("backbone_temporal_kernel_size", 3))
        _validate_temporal_kernel(temporal_kernel_size)
        source, feature_dim, provenance = _build_source(model_name, config)
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride_policy = "preserve"
        self.pooling = "adaptive_avg_3d"
        self.pretrained_provenance = provenance
        self.model = inflate_module_2d_to_3d(source, temporal_kernel_size)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(
                "A 3D video backbone expects clips shaped [B, T, C, H, W]; "
                f"received {tuple(clips.shape)}."
            )
        x = clips.transpose(1, 2).contiguous()
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        elif self.model_name.startswith("resnet"):
            features = self._forward_resnet(x)
        elif self.model_name == "densenet121":
            x = F.relu(self.model.features(x), inplace=True)
            features = torch.flatten(F.adaptive_avg_pool3d(x, (1, 1, 1)), 1)
        elif self.model_name in {"mobilenet_v3_large", "efficientnet_b0"}:
            features = torch.flatten(self.model.avgpool(self.model.features(x)), 1)
        elif self.model_name.startswith("convnext_"):
            features = torch.flatten(
                F.adaptive_avg_pool3d(self.model.features(x), (1, 1, 1)), 1
            )
            classifier_norm = self.model.classifier[0]
            layer_norm = getattr(classifier_norm, "norm", classifier_norm)
            features = F.layer_norm(
                features,
                layer_norm.normalized_shape,
                layer_norm.weight,
                layer_norm.bias,
                layer_norm.eps,
            )
        else:  # pragma: no cover - registry construction prevents this branch
            raise MLXUserError(f"No 3D feature path is registered for '{self.model_name}'.")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"3D backbone '{self.model_name}' returned {tuple(features.shape)}; "
                f"expected [B, {self.feature_dim}]."
            )
        return features

    def _forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        x = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        return torch.flatten(self.model.avgpool(x), 1)


class ResNet3DBackbone(InflatedImageBackbone3D):
    supported_model_names = ("resnet18", "resnet50")


class DenseNet1213DBackbone(InflatedImageBackbone3D):
    supported_model_names = ("densenet121",)


class MobileNetV3Large3DBackbone(InflatedImageBackbone3D):
    supported_model_names = ("mobilenet_v3_large",)


class EfficientNetB03DBackbone(InflatedImageBackbone3D):
    supported_model_names = ("efficientnet_b0",)


class ConvNeXt3DBackbone(InflatedImageBackbone3D):
    supported_model_names = (
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    )


class DraxNet3D(InflatedImageBackbone3D):
    supported_model_names = ("draxnet",)


class DraxMobileNetV3Large3D(InflatedImageBackbone3D):
    supported_model_names = ("drax_mobilenet_v3_large",)


BACKBONE_3D_REGISTRY: dict[str, type[InflatedImageBackbone3D]] = {
    alias: backbone_type
    for backbone_type in (
        ResNet3DBackbone,
        DenseNet1213DBackbone,
        MobileNetV3Large3DBackbone,
        EfficientNetB03DBackbone,
        ConvNeXt3DBackbone,
        DraxNet3D,
        DraxMobileNetV3Large3D,
    )
    for alias in backbone_type.supported_model_names
}


def build_spatiotemporal_backbone_3d(
    model_name: str, config: dict[str, Any]
) -> InflatedImageBackbone3D:
    from mlx.modes.image_classification.models import model_family_for

    if model_family_for(model_name) != "standard":
        raise MLXUserError(
            f"Model '{model_name}' is a one-shot/Siamese model and cannot be inflated to 3D."
        )
    backbone_type = BACKBONE_3D_REGISTRY.get(model_name)
    if backbone_type is None:
        supported = ", ".join(sorted(BACKBONE_3D_REGISTRY))
        raise MLXUserError(
            f"Model '{model_name}' has no 3D video-anomaly backbone. Available models: {supported}."
        )
    return backbone_type(model_name, config)


def _build_source(
    model_name: str, config: dict[str, Any]
) -> tuple[nn.Module, int, str]:
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
    return wrapper.adapter.model, int(wrapper.feature_dim), provenance


def _load_partial_resnet18_weights(model: nn.Module) -> None:
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


def _validate_temporal_kernel(value: int) -> None:
    if value < 1 or value % 2 == 0:
        raise MLXUserError("--backbone-temporal-kernel-size must be a positive odd integer.")


__all__ = [
    "BACKBONE_3D_REGISTRY",
    "ConvNeXt3DBackbone",
    "DenseNet1213DBackbone",
    "DraxMobileNetV3Large3D",
    "DraxNet3D",
    "EfficientNetB03DBackbone",
    "InflatedImageBackbone3D",
    "MobileNetV3Large3DBackbone",
    "ResNet3DBackbone",
    "TorchvisionConvNeXtBlock3D",
    "build_spatiotemporal_backbone_3d",
    "inflate_conv2d",
    "inflate_module_2d_to_3d",
]
