"""Authoritative built-in classifier construction metadata; no model imports."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class TorchvisionModelSpec:
    constructor: str
    weights: str
    stem: str
    head: str
    feature_family: str


TORCHVISION_MODELS = MappingProxyType({
    "resnet18": TorchvisionModelSpec("resnet18", "ResNet18_Weights", "conv1", "fc", "residual"),
    "resnet50": TorchvisionModelSpec("resnet50", "ResNet50_Weights", "conv1", "fc", "residual"),
    "densenet121": TorchvisionModelSpec("densenet121", "DenseNet121_Weights", "features.conv0", "classifier", "densenet"),
    "mobilenet_v3_large": TorchvisionModelSpec("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "features.0.0", "classifier.3", "mobile"),
    "efficientnet_b0": TorchvisionModelSpec("efficientnet_b0", "EfficientNet_B0_Weights", "features.0.0", "classifier.1", "mobile"),
    **{f"convnext_{size}": TorchvisionModelSpec(f"convnext_{size}", f"ConvNeXt_{size.title()}_Weights", "features.0.0", "classifier.2", "convnext")
       for size in ("tiny", "small", "base", "large")},
})
BUILTIN_BUILDERS = MappingProxyType({
    "draxnet": "mlx.modes.image_classification.models.draxnet:build_draxnet",
    "drax_mobilenet_v3_large": "mlx.modes.image_classification.models.drax_mobilenet:build_drax_mobilenet_v3_large",
})
BUILTIN_STANDARD_NAMES = frozenset(TORCHVISION_MODELS) | frozenset(BUILTIN_BUILDERS)
