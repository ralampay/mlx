import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_composition import run_adapter_action
from mlx.modes.object_detection.adapter_configuration import adapter_path
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
from mlx.modes.object_detection.adapter_slices import AdapterSliceRequest


@pytest.mark.parametrize('name,option', [
    ('model_path', '--checkpoint'), ('dataset_path', '--dataset'), ('output_path', '--output'),
])
@pytest.mark.parametrize('value', [None, '', '   '])
def test_experiment_requires_each_path(name, option, value):
    config = dict(model_path='foundation.pt', dataset_path='data', output_path='out')
    config[name] = value
    with pytest.raises(MLXUserError, match=option):
        AdapterExperimentRequest.from_config(config)


def test_unselected_parser_path_is_not_used():
    with pytest.raises(MLXUserError, match='--dataset'):
        adapter_path({'dataset_path': './tmp/dataset', '_explicit_options': set()},
                     'dataset_path', '--dataset')


def test_python_paths_are_resolved_without_mutating_configuration(tmp_path):
    config = {'dataset_path': tmp_path}
    assert adapter_path(config, 'dataset_path', '--dataset') == tmp_path.resolve()
    assert config == {'dataset_path': tmp_path}


def test_cached_slice_report_needs_only_output(tmp_path):
    request = AdapterSliceRequest.from_config({
        'action': 'adapter-slice-report', 'output_path': tmp_path,
    })
    assert request.checkpoint is None
    assert request.dataset is None
    request.validate(prediction=False)
    with pytest.raises(MLXUserError, match='Foundation checkpoint'):
        request.validate(prediction=True)


def test_verification_does_not_require_training_paths(tmp_path, monkeypatch):
    from mlx.modes.object_detection.libreyolo import adapter_backend
    calls = []

    class Verify:
        def __init__(self, model, checkpoint):
            calls.append((model, checkpoint))

        def execute(self):
            return object(), {'verified': True}

    monkeypatch.setattr(adapter_backend, 'VerifyFoundationCheckpoint', Verify)
    assert run_adapter_action({'action': 'adapter-verify', 'model': 'custom',
                               'model_path': tmp_path / 'model.pt'}) == {'verified': True}
    assert calls == [('custom', tmp_path / 'model.pt')]


def test_report_does_not_require_training_paths(tmp_path, monkeypatch):
    from mlx.modes.object_detection import adapter_report

    class Report:
        def __init__(self, output, **kwargs):
            assert output == tmp_path

        def execute(self):
            return {'reported': True}

    monkeypatch.setattr(adapter_report, 'GenerateAdapterReport', Report)
    assert run_adapter_action({'action': 'adapter-report', 'output_path': tmp_path}) == {'reported': True}


def test_prepare_passes_explicit_paths_and_seed(tmp_path, monkeypatch):
    from mlx.modes.object_detection import adapter_data
    calls = []

    class Prepare:
        def __init__(self, source, destination, *, seed):
            calls.append((source, destination, seed))

        def execute(self):
            return {'prepared': True}

    monkeypatch.setattr(adapter_data, 'PrepareDawnAdapterDataset', Prepare)
    config = {'action': 'adapter-prepare', 'dataset_path': tmp_path / 'source',
              'output_path': tmp_path / 'destination', 'random_seed': 0}
    assert run_adapter_action(config) == {'prepared': True}
    assert calls == [(tmp_path / 'source', tmp_path / 'destination', 0)]


@pytest.mark.parametrize('classes,selection,error', [
    (['widget'], 'changed', 'selection differs'),
    (['different'], 'split', 'class order differs'),
])
def test_slice_prediction_checks_supplied_foundation_taxonomy(tmp_path, monkeypatch, classes, selection, error):
    import json
    from mlx.modes.object_detection import adapter_slices

    checkpoint = tmp_path / 'foundation.pt'
    checkpoint.touch()
    dataset = tmp_path / 'data'
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (dataset / kind / split).mkdir(parents=True)
    (dataset / 'manifest.json').write_text(json.dumps({
        'classes': classes, 'selection_sha256': selection,
    }))
    (dataset / 'data.yaml').write_text(json.dumps({'names': classes}))
    (tmp_path / 'study.json').write_text(json.dumps({'dataset_selection_sha256': 'split'}))

    class Verify:
        def __init__(self, *args):
            pass

        def execute(self):
            return object(), {'classes': {0: 'widget'}, 'nc': 1}

    class Backend:
        def prepare(self, request, model_name):
            return Verify().execute()[1], request.device
    request = AdapterSliceRequest(tmp_path, checkpoint, dataset)
    with pytest.raises(MLXUserError, match=error):
        adapter_slices.CacheAdapterSlicePredictions(request, backend=Backend()).execute()
