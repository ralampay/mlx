"""Portable adapter extension and injection contracts."""
import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from mlx.modes.object_detection.feature_adapters import (
    DEFAULT_FEATURE_ADAPTER_REGISTRY, FeatureAdapterDefinition, FeatureAdapterRegistry,
    adapter_state_dict, available_adapters, create_adapter, inject_adapters,
    load_adapter_state_dict,
)


class ScaleFeature(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, value):
        return value * self.scale


def registry():
    return DEFAULT_FEATURE_ADAPTER_REGISTRY.register(
        'scale-example', FeatureAdapterDefinition(ScaleFeature, parameters=()))


def test_custom_adapter_roundtrip_and_registry_isolation():
    selected = registry()
    assert 'scale-example' in available_adapters(registry=selected)
    assert 'scale-example' not in available_adapters()
    assert isinstance(create_adapter('scale-example', 3, registry=selected), ScaleFeature)
    with pytest.raises(TypeError):
        selected.entries['other'] = selected.resolve('scale-example')
    base = nn.Sequential(nn.Conv2d(3, 3, 1))
    model, restored = copy.deepcopy(base), copy.deepcopy(base)
    sample = torch.randn(2, 3, 4, 4)
    inject_adapters(model, 'scale-example', {'0': 3}, registry=selected)
    torch.testing.assert_close(model(sample), base(sample), rtol=0, atol=0)
    model(sample).square().mean().backward()
    assert model[0].adapter.scale.grad is not None
    assert model[0].base.weight.grad is None
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1).step()
    inject_adapters(restored, 'scale-example', {'0': 3}, registry=selected)
    load_adapter_state_dict(restored, adapter_state_dict(model))
    torch.testing.assert_close(model(sample), restored(sample), rtol=0, atol=0)
    with pytest.raises(ValueError, match='mismatch'):
        load_adapter_state_dict(restored, {})


def test_failed_construction_restores_model_and_trainability():
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Conv2d(3, 3, 1))
    model[1].bias.requires_grad_(False)
    original = list(model.children())
    flags = [p.requires_grad for p in model.parameters()]
    calls = []
    def build(base):
        base.requires_grad_(False)
        calls.append(base)
        if len(calls) == 2:
            raise ValueError('construction failed')
        return nn.Sequential(base, nn.Identity())
    selected = FeatureAdapterRegistry().register('broken', FeatureAdapterDefinition(build, 'conv', ()))
    with pytest.raises(ValueError, match='construction failed'):
        inject_adapters(model, 'broken', {'0': 3, '1': 3}, registry=selected)
    assert list(model.children()) == original
    assert [p.requires_grad for p in model.parameters()] == flags


@pytest.mark.parametrize('targets', [{'0': 3, '0.child': 3}, {'missing': 3}, {'': 3}, {'0': 0}])
def test_invalid_targets_leave_model_unchanged(targets):
    model = nn.Sequential(nn.Conv2d(3, 3, 1))
    child = model[0]
    with pytest.raises(ValueError):
        inject_adapters(model, 'scale-example', targets, registry=registry())
    assert model[0] is child and all(p.requires_grad for p in model.parameters())


def test_missing_trainable_path_is_validated_before_injection():
    model = nn.Sequential(nn.Conv2d(3, 3, 1))
    child = model[0]
    with pytest.raises(ValueError, match='path'):
        inject_adapters(model, 'lora', {'0': 3}, train_head=True)
    assert model[0] is child and all(p.requires_grad for p in model.parameters())


def test_explicit_trainable_modules_without_detector_head():
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Conv2d(3, 2, 1))
    inject_adapters(model, 'scale-example', {'0': 3}, registry=registry(), trainable_modules=('1',))
    assert model[1].weight.requires_grad
    assert not model[0].base.weight.requires_grad
    with pytest.raises(ValueError, match='already'):
        inject_adapters(model, 'scale-example', {'0': 3}, registry=registry())


def test_alias_targets_rejected():
    child = nn.Conv2d(3, 3, 1)
    model = nn.ModuleDict({'one': child, 'two': child})
    with pytest.raises(ValueError, match='alias'):
        inject_adapters(model, 'lora', {'one': 3, 'two': 3})


def test_custom_convolution_reconstruction_uses_injected_registry():
    from mlx.modes.object_detection.libreyolo.adapter_loading import ApplyYOLOXAdapter
    selected = DEFAULT_FEATURE_ADAPTER_REGISTRY.register('custom-conv', replace(
        DEFAULT_FEATURE_ADAPTER_REGISTRY.resolve('lora')))
    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleDict({'lateral_conv0': nn.Sequential(nn.Conv2d(3, 3, 1))})
            self.head = nn.Identity()
    first = Detector()
    second = copy.deepcopy(first)
    path = 'backbone.lateral_conv0.0'
    inject_adapters(first, 'custom-conv', {path: 3}, registry=selected)
    config = {'method': 'custom-conv', 'injected_modules': [path]}
    ApplyYOLOXAdapter(second, config, adapter_state_dict(first), registry=selected).execute()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('name', ['', '../external', 'UPPER', 'frozen', 'head-only', 'full-finetune'])
def test_registry_rejects_invalid_or_reserved_names(name):
    with pytest.raises(ValueError, match='names'):
        FeatureAdapterRegistry().register(name, FeatureAdapterDefinition(ScaleFeature, parameters=()))


def test_unknown_adapter_and_shape_mismatch():
    model = nn.Sequential(nn.Conv2d(3, 3, 1))
    with pytest.raises(ValueError, match='Unknown'):
        inject_adapters(model, 'unregistered', {'0': 3})
    inject_adapters(model, 'scale-example', {'0': 3}, registry=registry())
    state = adapter_state_dict(model)
    key = next(iter(state))
    state[key] = torch.zeros(9)
    before = model[0].adapter.scale.detach().clone()
    with pytest.raises(ValueError, match='shape'):
        load_adapter_state_dict(model, state)
    torch.testing.assert_close(model[0].adapter.scale, before)


def test_slice_validation_accepts_registered_methods(tmp_path):
    from mlx.modes.object_detection.adapter_slices import AdapterSliceRequest
    from mlx.core.exceptions import MLXUserError
    request = AdapterSliceRequest(tmp_path, tmp_path/'model', tmp_path/'data',
                                  methods=('scale-example',), comparison_method='scale-example')
    request.validate(prediction=False, registry=registry())
    with pytest.raises(MLXUserError, match='Unknown comparison'):
        request.validate(prediction=False)
