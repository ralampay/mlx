from __future__ import annotations

from collections.abc import Callable
import inspect

from torch import nn

from mlx.core.exceptions import MLXUserError

CUSTOM_STANDARD_MODEL_BUILDERS: dict[str, Callable] = {}

SUPPORTED_TORCHVISION_MODELS = {
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_tiny",
}


def register_standard_model(name: str, builder) -> None:
    CUSTOM_STANDARD_MODEL_BUILDERS[name] = builder


def registered_standard_model_names() -> list[str]:
    return sorted(CUSTOM_STANDARD_MODEL_BUILDERS.keys())


def build_standard_model(
    model_name: str,
    *,
    num_classes: int,
    colored: bool,
    pretrained: bool,
    config: dict | None = None,
):
    custom_builder = CUSTOM_STANDARD_MODEL_BUILDERS.get(model_name)
    if custom_builder is not None:
        builder_params = {
            "num_classes": num_classes,
            "colored": colored,
            "pretrained": pretrained,
        }
        if "config" in inspect.signature(custom_builder).parameters:
            builder_params["config"] = config or {}
        return custom_builder(**builder_params)

    if model_name in SUPPORTED_TORCHVISION_MODELS:
        try:
            from torchvision import models as torchvision_models
        except ImportError as exc:
            raise MLXUserError(
                "Torchvision is required for standard classification models. Install it with 'pip install torchvision'."
            ) from exc

        model, stem_attr = _build_torchvision_model(
            model_name=model_name,
            torchvision_models=torchvision_models,
            pretrained=pretrained,
        )

        if not colored:
            _replace_stem_conv(model, stem_attr)

        _replace_classifier_head(model, model_name, num_classes)
        return model

    raise MLXUserError(f"Unsupported standard image-classification model '{model_name}'.")


def _build_torchvision_model(*, model_name: str, torchvision_models, pretrained: bool):
    if model_name == "resnet18":
        weights = torchvision_models.ResNet18_Weights.DEFAULT if pretrained else None
        return torchvision_models.resnet18(weights=weights), "conv1"
    if model_name == "resnet50":
        weights = torchvision_models.ResNet50_Weights.DEFAULT if pretrained else None
        return torchvision_models.resnet50(weights=weights), "conv1"
    if model_name == "densenet121":
        weights = torchvision_models.DenseNet121_Weights.DEFAULT if pretrained else None
        return torchvision_models.densenet121(weights=weights), "features.conv0"
    if model_name == "mobilenet_v3_large":
        weights = torchvision_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return torchvision_models.mobilenet_v3_large(weights=weights), "features.0.0"
    if model_name == "efficientnet_b0":
        weights = torchvision_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return torchvision_models.efficientnet_b0(weights=weights), "features.0.0"
    if model_name == "convnext_tiny":
        weights = torchvision_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        return torchvision_models.convnext_tiny(weights=weights), "features.0.0"
    if model_name == "convnext_small":
        weights = torchvision_models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        return torchvision_models.convnext_small(weights=weights), "features.0.0"
    if model_name == "convnext_base":
        weights = torchvision_models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        return torchvision_models.convnext_base(weights=weights), "features.0.0"
    if model_name == "convnext_large":
        weights = torchvision_models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
        return torchvision_models.convnext_large(weights=weights), "features.0.0"

    raise MLXUserError(f"Unsupported standard image-classification model '{model_name}'.")


def _replace_stem_conv(model, stem_attr: str) -> None:
    import torch

    original_conv = _resolve_module_attr(model, stem_attr)
    replacement_conv = nn.Conv2d(
        1,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    with torch.no_grad():
        replacement_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        if original_conv.bias is not None and replacement_conv.bias is not None:
            replacement_conv.bias.copy_(original_conv.bias)
    _assign_module_attr(model, stem_attr, replacement_conv)


def _replace_classifier_head(model, model_name: str, num_classes: int) -> None:
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    if model_name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return
    if model_name == "mobilenet_v3_large":
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            model.classifier[2],
            nn.Linear(model.classifier[3].in_features, num_classes),
        )
        return
    if model_name == "efficientnet_b0":
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
        return
    if model_name.startswith("convnext_"):
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return
    raise MLXUserError(f"Unsupported standard image-classification model '{model_name}'.")


def _resolve_module_attr(model, attr_path: str):
    module = model
    for part in attr_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _assign_module_attr(model, attr_path: str, new_module) -> None:
    parent_path, _, leaf = attr_path.rpartition(".")
    parent = _resolve_module_attr(model, parent_path) if parent_path else model
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)
