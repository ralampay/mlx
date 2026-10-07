# Migrated from project-owned LibreYOLO adapters; retained MIT provenance.
"""Checkpoint-independent adapters for convolutional feature maps."""

from .layers import create_adapter, available_adapters
from .injection import inject_adapters, count_parameters, adapter_state_dict, load_adapter_state_dict
from .residual_fusion import DraxResidualFusionConv2d
from .hybrid import DraxHybridConv2d, DraxSpatialConv2d

__all__ = ["create_adapter", "available_adapters", "inject_adapters", "count_parameters", "adapter_state_dict", "load_adapter_state_dict", "DraxHybridConv2d", "DraxSpatialConv2d", "DraxResidualFusionConv2d"]

from .registry import FeatureAdapterDefinition, FeatureAdapterRegistry, DEFAULT_FEATURE_ADAPTER_REGISTRY

__all__ += ["FeatureAdapterDefinition", "FeatureAdapterRegistry", "DEFAULT_FEATURE_ADAPTER_REGISTRY"]
