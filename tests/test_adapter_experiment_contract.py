"""Workflow behavior independent of a detector provider or a real dataset."""
import builtins
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest, RunAdapterExperiment
from mlx.modes.object_detection.feature_adapters import FeatureAdapterDefinition, FeatureAdapterRegistry


class Backend:
    def __init__(self):
        self.calls = []
    def validate(self, request):
        self.calls.append('validate')
    def study_metadata(self, request):
        return {}
    def resolve_device(self, requested):
        return requested
    def verify_foundation(self, request):
        return None, {'classes': {0: 'object'}, 'nc': 1, 'sha256': 'foundation'}
    def collect_environment(self, request, device):
        return {}
    def run_condition(self, request, method, seed, dataset, foundation, device, registry):
        self.calls.append((method, seed, registry))
        metrics = dict(status='completed', method=method, seed=seed,
            checkpoint_sha256='foundation', dataset_selection_sha256='data',
            physical_batch_size=1, effective_batch_size=1, image_size=17,
            device='example', amp=True, epochs=0 if method == 'frozen' else request.epochs,
            learning_rate=request.lr, adapter_rank=None, adapter_reduction=None,
            adapter_alpha=None if method == 'frozen' else request.alpha,
            train_head=False, adapter_target=None if method == 'frozen' else request.target)
        path = request.output / method / f'seed-{seed}' / 'metrics.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(metrics))
        return metrics
    def release(self, device):
        pass


def setup(tmp_path):
    request = AdapterExperimentRequest('example-model', tmp_path/'model', tmp_path/'data',
        tmp_path/'out', ('example',), (2,), image_size=17, device='example')
    registry = FeatureAdapterRegistry().register('example', FeatureAdapterDefinition(lambda n: None, parameters=()))
    data = {'classes': ['object'], 'dataset': 'synthetic', 'selection_sha256': 'data'}
    return request, registry, data


def test_provider_free_workflow_and_resume(tmp_path, monkeypatch):
    request, registry, data = setup(tmp_path)
    original = builtins.__import__
    def isolated(name, *args, **kwargs):
        if 'libreyolo' in name or 'ultralytics' in name:
            raise AssertionError('provider import in generic workflow')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', isolated)
    backend = Backend()
    command = RunAdapterExperiment(request, backend=backend, registry=registry, dataset_loader=lambda p: data)
    rows = command.execute()
    assert len(rows) == 2
    assert backend.calls[1:] == [('frozen', 2, registry), ('example', 2, registry)]
    assert command.execute() == rows
    assert len(backend.calls) == 4
    changed = replace(request, alpha=2)
    with pytest.raises(MLXUserError, match='incompatible'):
        RunAdapterExperiment(changed, backend=backend, registry=registry, dataset_loader=lambda p: data).execute()


def test_class_mismatch_before_output_creation(tmp_path):
    request, registry, data = setup(tmp_path)
    data['classes'] = ['different']
    with pytest.raises(MLXUserError, match='class order'):
        RunAdapterExperiment(request, backend=Backend(), registry=registry, dataset_loader=lambda p: data).execute()
    assert not request.output.exists()


def test_backend_failure_stops_following_conditions(tmp_path):
    request, registry, data = setup(tmp_path)
    backend = Backend()
    def fail(*args):
        backend.calls.append('failed')
        raise MLXUserError('backend failed')
    backend.run_condition = fail
    with pytest.raises(MLXUserError, match='backend failed'):
        RunAdapterExperiment(request, backend=backend, registry=registry, dataset_loader=lambda p: data).execute()
    assert backend.calls == ['validate', 'failed']


def test_zero_shot_provenance_uses_injected_collector(tmp_path):
    from mlx.modes.object_detection.zero_shot.workflow import RunTransferStudy
    provenance = {'source_revision': ['source'], 'evaluation_signature': {'implementation': 'hash'}}
    command = RunTransferStudy({}, tmp_path, object(), object(),
        provenance_collector=SimpleNamespace(execute=lambda: provenance))
    command._provenance()
    assert command.source_revision == provenance['source_revision']
    assert command.evaluation_signature == provenance['evaluation_signature']


@pytest.mark.parametrize('content', ['[]', '{', '{"classes": [["nested"]]}', '{"classes": [""]}'])
def test_malformed_dataset_manifest_is_a_user_error(tmp_path, content):
    from mlx.modes.object_detection.prepared_adapter_data import load_prepared_adapter_dataset
    (tmp_path/'manifest.json').write_text(content)
    (tmp_path/'data.yaml').write_text('names: [object]')
    with pytest.raises(MLXUserError):
        load_prepared_adapter_dataset(tmp_path)


def test_prepared_dataset_uses_manifest_taxonomy(tmp_path):
    from mlx.modes.object_detection.prepared_adapter_data import load_prepared_adapter_dataset
    (tmp_path/'manifest.json').write_text(json.dumps({'classes': ['example']}))
    (tmp_path/'data.yaml').write_text('names: [example]')
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (tmp_path/kind/split).mkdir(parents=True)
    assert load_prepared_adapter_dataset(tmp_path)['classes'] == ['example']
    (tmp_path/'data.yaml').write_text('names: [different]')
    with pytest.raises(MLXUserError, match='class order'):
        load_prepared_adapter_dataset(tmp_path)


def test_installed_source_snapshot_is_scoped_and_content_addressed(tmp_path):
    from mlx.modes.object_detection.libreyolo.transfer_provenance import CollectTransferProvenance
    package = tmp_path/'package'
    package.mkdir()
    (package/'module.py').write_text('value = 1\n')
    (package/'__pycache__').mkdir()
    (package/'__pycache__/module.pyc').write_bytes(b'cache')
    collector = CollectTransferProvenance(tmp_path/'out', 'cpu')
    destination = collector.output/'sources'
    first = collector._snapshot_package('example', package, destination, {'version': '1'})
    assert first == collector._snapshot_package('example', package, destination, {'version': '1'})
    manifest = json.loads((collector.output/first['directory']/'manifest.json').read_text())
    assert set(manifest['files']) == {'module.py'}
    (package/'module.py').write_text('value = 2\n')
    assert collector._snapshot_package('example', package, destination, {'version': '1'})['revision'] != first['revision']


def test_installed_vcs_metadata_does_not_use_enclosing_repository(tmp_path, monkeypatch):
    import importlib.metadata
    from mlx.modes.object_detection.libreyolo.adapter_backend import _git_commit
    enclosing = tmp_path/'checkout'
    (enclosing/'.git').mkdir(parents=True)
    installed = enclosing/'environment/package'
    installed.mkdir(parents=True)
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: SimpleNamespace(
        read_text=lambda filename: json.dumps({'vcs_info': {'commit_id': 'package-commit'}})))
    assert _git_commit(installed, distribution='example') == 'package-commit'
    assert _git_commit(installed) is None


def test_provider_condition_executes_custom_adapter_with_synthetic_training(tmp_path, monkeypatch):
    import copy
    import sys
    import torch
    from torch import nn
    from mlx.modes.object_detection.feature_adapters import DEFAULT_FEATURE_ADAPTER_REGISTRY
    from mlx.modes.object_detection.libreyolo import adapter_execution as integration
    from mlx.modes.object_detection.libreyolo.adapter_loading import ApplyYOLOXAdapter

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleDict({'lateral_conv0': nn.Sequential(nn.Conv2d(3, 3, 1))})
            self.head = nn.Identity()
        def forward(self, x):
            return self.head(self.backbone['lateral_conv0'](x))

    original = Model()
    info = {'sha256': 'foundation', 'checkpoint': 'synthetic', 'nc': 1, 'classes': {0: 'object'}}
    monkeypatch.setattr(integration, 'VerifyFoundationCheckpoint',
        lambda *a: SimpleNamespace(execute=lambda: (copy.deepcopy(original), info)))
    monkeypatch.setitem(sys.modules, 'libreyolo.utils.serialization', SimpleNamespace(
        load_untrusted_torch_file=lambda path, **kw: torch.load(path, weights_only=True, **kw)))

    class Wrapper:
        def __init__(self, model):
            self.model = model
        def train(self, **options):
            optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.1)
            self.model(torch.ones(2, 3, 8, 8)).square().mean().backward()
            optimizer.step()
            state = self.model.state_dict()
            checkpoint = tmp_path/'selected.pt'
            torch.save({'model': state, 'train_model': state, 'best_epoch': 1}, checkpoint)
            return {'best_checkpoint': str(checkpoint), 'last_checkpoint': str(checkpoint),
                    'epoch_metrics': [{'epoch': 1, 'train_loss': 1.0}]}
        def val(self, **options):
            return {'metrics/mAP50': .3, 'metrics/mAP50-95': .2, 'precision': .5, 'recall': .4}

    monkeypatch.setattr(integration, 'build_experimental_yolox', lambda model, *a: Wrapper(model))
    monkeypatch.setattr(integration, 'measure_precision_recall', lambda *a, **kw: {'precision': .5, 'recall': .4})
    registry = DEFAULT_FEATURE_ADAPTER_REGISTRY.register('custom-conv',
        DEFAULT_FEATURE_ADAPTER_REGISTRY.resolve('lora'))
    request = AdapterExperimentRequest('yolox-n', tmp_path/'foundation', tmp_path/'dataset',
        tmp_path/'output', ('custom-conv',), (1,), image_size=32, epochs=1, device='cpu')
    dataset = {'dataset': 'synthetic', 'selection_sha256': 'data',
               'splits': {split: {'images': 2} for split in ('train', 'val', 'test')}}
    result = integration.RunLibreYOLOAdapterCondition(request, 'custom-conv', 1, dataset,
        info, torch.device('cpu'), registry=registry).execute()
    assert result['status'] == 'completed' and result['trainable_parameters_changed']
    assert result['foundation_parameters_unchanged'] and result['adapter_rank'] == request.rank
    checkpoint = request.output/'custom-conv/seed-1/adapter/checkpoint.pt'
    payload = torch.load(checkpoint, weights_only=True)
    restored = ApplyYOLOXAdapter(copy.deepcopy(original), payload['config'], payload['state'], registry=registry).execute()
    assert any('b.weight' in key for key in restored.state_dict())


def test_custom_registry_reaches_calibration(tmp_path, monkeypatch):
    from mlx.modes.object_detection.feature_adapters import DEFAULT_FEATURE_ADAPTER_REGISTRY
    from mlx.modes.object_detection.libreyolo import adapter_execution as integration, adapter_backend
    selected = DEFAULT_FEATURE_ADAPTER_REGISTRY.register('custom-conv',
        DEFAULT_FEATURE_ADAPTER_REGISTRY.resolve('lora'))
    request = AdapterExperimentRequest('yolox-n', tmp_path, tmp_path, tmp_path,
                                       ('custom-conv',), (1,))
    calls = {}
    monkeypatch.setattr(integration.LibreYOLOExperimentBackend, 'resolve_device', lambda *a: 'device')
    monkeypatch.setattr(integration.LibreYOLOExperimentBackend, 'collect_environment', lambda *a: {})
    def calibrate(*args, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(execute=lambda: {'batch': 1})
    monkeypatch.setattr(adapter_backend, 'CalibrateAdapterBatchSize', calibrate)
    assert integration.CalibrateAdapterExperiment(request, registry=selected).execute() == {'batch': 1}
    assert calls['registry'] is selected and calls['profiles'] == ('custom-conv',)


def test_provider_construction_failure_has_context_and_preserves_cause(monkeypatch):
    from mlx.modes.object_detection.libreyolo import adapter_execution as integration
    def fail(*args, **kwargs):
        raise ValueError('invalid adapter factory')
    monkeypatch.setattr(integration.RunLibreYOLOAdapterCondition, 'execute', fail)
    with pytest.raises(MLXUserError, match="adapter 'example'.*seed 3") as error:
        integration.LibreYOLOExperimentBackend().run_condition(None, 'example', 3, {}, {}, None, None)
    assert isinstance(error.value.__cause__, ValueError)
