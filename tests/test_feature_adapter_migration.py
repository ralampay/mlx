"""Legacy API parity and self-contained model compatibility."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection import feature_adapters as current
from mlx.modes.object_detection.libreyolo.adapter_checkpoint import (
    FORMAT, LoadAdapterDetector, select_validation_candidate,
)
from mlx.modes.object_detection.libreyolo.adapter_targets import yolox_targets


@pytest.mark.parametrize('method', current.available_adapters())
def test_legacy_parity_initialization_gradient_state_and_update(method):
    from libreyolo import adapters as legacy
    torch.manual_seed(7)
    original = nn.Sequential(nn.Conv2d(16, 16, 1))
    models = [copy.deepcopy(original), copy.deepcopy(original)]
    for api, model in zip((legacy, current), models):
        torch.manual_seed(12)
        api.inject_adapters(model, method, {'0': 16}, reduction=4, rank=8)
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key])
    x = torch.randn(2, 16, 7, 7)
    for model in models:
        assert torch.equal(model(x), original(x))
        model(x).square().mean().backward()
    for (a, p), (b, q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert a == b and p.requires_grad == q.requires_grad
        if p.requires_grad:
            assert torch.equal(p.grad, q.grad)
        else:
            assert p.grad is None and q.grad is None
    before = {k: v.clone() for k, v in models[1].named_parameters()}
    torch.optim.SGD([p for p in models[1].parameters() if p.requires_grad], lr=.1).step()
    assert any(not torch.equal(before[k], p) for k,p in models[1].named_parameters() if p.requires_grad)
    assert all(torch.equal(before[k], p) for k,p in models[1].named_parameters() if not p.requires_grad)
    legacy.load_adapter_state_dict(models[0], current.adapter_state_dict(models[1]))
    torch.testing.assert_close(models[0](x), models[1](x), rtol=0, atol=0)


def test_full_detector_load_without_foundation(tmp_path):
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    model = LibreYOLOXModel(config='n', nb_classes=6).eval()
    targets = yolox_targets(model, 'neck', 'drax-residual-fusion')
    current.inject_adapters(model, 'drax-residual-fusion', targets, reduction=16)
    path = tmp_path / 'detector.pt'
    config = {'method': 'drax-residual-fusion', 'injected_modules': list(targets),
              'adapter_reduction': 16, 'foundation_checkpoint': '/missing/foundation.pt'}
    torch.save({'format': FORMAT, 'size': 'n', 'names': ['a','b','c','d','e','f'],
                'config': config, 'model': model.state_dict()}, path)
    restored = LoadAdapterDetector(path).execute()
    x = torch.randn(1,3,64,64)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored.model(x), rtol=0, atol=0)
    torch.save({'format': 'not-a-detector'}, path)
    with pytest.raises(MLXUserError, match='Expected'):
        LoadAdapterDetector(path).execute()


def test_selection_uses_validation_not_test_and_breaks_ties():
    candidates = [SimpleNamespace(seed=s, config={'selected_validation_mAP50_95': val},
                                 metrics={'mAP50_95': test})
                  for s,val,test in [(3,.4,.9),(2,.5,.1),(1,.5,.2)]]
    assert select_validation_candidate(candidates).seed == 1
    with pytest.raises(MLXUserError, match='finite'):
        select_validation_candidate([])


def test_active_adapter_package_does_not_import_legacy_provider():
    from pathlib import Path
    for path in Path(current.__file__).parent.glob('*.py'):
        assert 'from libreyolo' not in path.read_text()
        assert 'import libreyolo' not in path.read_text()
