from __future__ import annotations

from torch import nn

from mlx.core.exceptions import MLXUserError

CUSTOM_STANDARD_MODEL_BUILDERS: dict[str, callable] = {}


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
):
    custom_builder = CUSTOM_STANDARD_MODEL_BUILDERS.get(model_name)
    if custom_builder is not None:
        return custom_builder(
            num_classes=num_classes,
            colored=colored,
            pretrained=pretrained,
        )

    if model_name in {"resnet18", "resnet50"}:
        try:
            from torchvision import models as torchvision_models
        except ImportError as exc:
            raise MLXUserError(
                "Torchvision is required for resnet models. Install it with 'pip install torchvision'."
            ) from exc

        if model_name == "resnet18":
            weights = torchvision_models.ResNet18_Weights.DEFAULT if pretrained else None
            model = torchvision_models.resnet18(weights=weights)
        else:
            weights = torchvision_models.ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision_models.resnet50(weights=weights)

        if not colored:
            import torch

            original_conv = model.conv1
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
            model.conv1 = replacement_conv

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise MLXUserError(f"Unsupported standard image-classification model '{model_name}'.")
