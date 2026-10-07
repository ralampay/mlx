import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.core.partitions import partition_rows, partition_hash
from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
from mlx.modes.autoencoder.architectures.structured import SpectralAutoencoderDefinition, OrderedAutoencoderDefinition
from mlx.modes.autoencoder.commands import TrainAutoencoder
from mlx.modes.autoencoder.data import EmbeddingCsvLoader
from mlx.modes.autoencoder.objectives import OrderedReconstructionObjective, build_objective, validate_training_variant
from mlx.modes.autoencoder.regularization import CovarianceLoss, LeastVolumeLoss
from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
from mlx.modes.text_embedding.compression_controls import FitPcaVectors, PcaVectors, TruncateVectors
from mlx.modes.text_embedding.experiment_config import load_experiment_config
from mlx.modes.text_embedding.experiment_templates import pilot_config, confirmation_config


def model_config():
    return {"input_dimensions": 4, "hidden_dimensions": 4, "bottleneck_dimensions": 3, "prefix_dimensions": [1, 2, 3]}


def test_penalties_match_formulas_and_gradients():
    z = torch.tensor([[1., 3., 2.], [4., 5., 8.], [7., 6., 2.]], requires_grad=True)
    centered = z - z.mean(0)
    covariance = centered.T @ centered / 2
    expected = sum(covariance[i, j] ** 2 for i in range(3) for j in range(3) if i != j) / 6
    penalty = CovarianceLoss().penalty(z)
    assert torch.allclose(penalty, expected)
    lv = LeastVolumeLoss().penalty(z)
    assert torch.allclose(lv, torch.prod(z.std(0) + 1e-3) ** (1 / 3))
    (penalty + lv).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert CovarianceLoss().penalty(torch.ones(3, 2)) == 0
    assert LeastVolumeLoss().penalty(torch.ones(3, 2)) == pytest.approx(1e-3)
    for loss in (CovarianceLoss(), LeastVolumeLoss()):
        with pytest.raises(MLXUserError, match="at least two"):
            loss.penalty(torch.ones(1, 2))
    for rho in (-1, float("nan"), True):
        with pytest.raises(MLXUserError):
            LeastVolumeLoss(rho=rho)


def test_calibration_components_zero_and_capabilities():
    model = SpectralAutoencoderDefinition().build(model_config())
    values = torch.randn(12, 4)
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss = LeastVolumeLoss(rho=.1)
    loss.calibrate(model, values)
    assert loss.scale == pytest.approx(loss.calibration["reconstruction"] / loss.calibration["penalty"])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    z = model.encode(values)
    total, components = loss.components(model.decode(z), values, latent=z)
    assert torch.allclose(total, components["reconstruction"] + components["weighted_penalty"])
    prediction = torch.randn(1, 4)
    for criterion in (LeastVolumeLoss(rho=0), CovarianceLoss(rho=0)):
        assert torch.equal(criterion(prediction, values[:1]), nn.functional.mse_loss(prediction, values[:1]))
    with pytest.raises(MLXUserError, match="constrained"):
        build_objective(nn.Linear(4, 4), LeastVolumeLoss(), 1)


def test_ordered_masks_rng_and_validation():
    class Identity:
        def encode(self, values): return values
        def decode(self, values): return values
    x = torch.ones(4, 3)
    first = OrderedReconstructionObjective((1, 2, 3), 42)
    second = OrderedReconstructionObjective((1, 2, 3), 42)
    torch.manual_seed(17)
    state = torch.random.get_rng_state()
    a = [first.evaluate(Identity(), x, training=True).loss for _ in range(20)]
    b = [second.evaluate(Identity(), x, training=True).loss for _ in range(20)]
    assert a == b and torch.equal(state, torch.random.get_rng_state())
    assert first.evaluate(Identity(), x, training=False).loss == pytest.approx(1 / 3)
    assert first.evaluate(Identity(), x, training=False).loss == first.evaluate(Identity(), x, training=False).loss
    model = OrderedAutoencoderDefinition().build(model_config())
    with pytest.raises(MLXUserError, match="MSE only"):
        build_objective(model, nn.L1Loss(), 0)


@pytest.mark.parametrize('model,loss,options', [
    ('simple-spectral', 'mse-least-volume', {}), ('simple', 'mse-covariance', {}),
    ('ordered-simple', 'mse', {'prefix_dimensions': [1, 2, 3]}),
])
def test_train_checkpoint_reload_and_split(tmp_path, model, loss, options):
    source = tmp_path / 'vectors.csv'
    with source.open('w') as f:
        writer = csv.writer(f); writer.writerow(['id', 'embedding'])
        writer.writerows((str(i), json.dumps(row)) for i, row in enumerate(np.random.default_rng(9).normal(size=(15, 4)).tolist()))
    result = TrainAutoencoder(AutoencoderTrainRequest(model=model, loss=loss, input_path=str(source),
        output_path=str(tmp_path / 'train'), autoencoder_config=options, hidden_dim=4, bottleneck_dim=3,
        epochs=2, batch_size=4, minimum_batch_size=2, plots=False)).execute()
    full = AutoencoderRepresentationTransformer(result.checkpoint_path)
    values = [[1., 2., 3., 4.]]
    assert full.transform(values) == full.transform(values)
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    train, val = partition_rows(15, .2, 42)
    assert checkpoint['split_hash'] == partition_hash(train, val)
    assert checkpoint['validation_components']['reconstruction'] > 0
    if model == 'ordered-simple':
        prefix = AutoencoderRepresentationTransformer(result.checkpoint_path, output_dimensions=2)
        assert prefix.transform(values)[0] == pytest.approx(full.transform(values)[0][:2])
    else:
        with pytest.raises(MLXUserError, match='prefix'):
            AutoencoderRepresentationTransformer(result.checkpoint_path, output_dimensions=2)
    if model == 'simple-spectral':
        assert any('parametrizations.weight' in k for k in checkpoint['state_dict'])
        assert checkpoint['calibration']['scale'] > 0


def test_pca_training_partition_and_controls(tmp_path):
    values = np.random.default_rng(8).normal(size=(10, 4))
    train, val = partition_rows(10, .2, 42)
    FitPcaVectors(values, train, 3, tmp_path / 'pca', split_hash=partition_hash(train, val)).execute()
    original = PcaVectors(tmp_path / 'pca/pca.npz', 2)
    changed = values.copy(); changed[val] += 10000
    FitPcaVectors(changed, train, 3, tmp_path / 'other', split_hash=partition_hash(train, val)).execute()
    other = PcaVectors(tmp_path / 'other/pca.npz', 2)
    assert np.allclose(original.transform(values), other.transform(values))
    assert np.allclose(original.mean, values[train].mean(0))
    assert np.array_equal(TruncateVectors(4, 2).transform(values), values.astype(np.float32)[:, :2])


def test_duplicate_variants_are_rejected(tmp_path):
    config = tiny_config('pilot')
    config['variants'].append(config['variants'][0])
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(MLXUserError, match='unique'):
        load_experiment_config(path)


def tiny_config(phase):
    data = pilot_config() if phase == 'pilot' else confirmation_config({'mse-least-volume': .1, 'mse-covariance': .1})
    data.update(datasets=['one', 'two'], seeds=[42, 43], training={
        'hidden_dim': 4, 'epochs': 1, 'batch_size': 4, 'lr': .001, 'val_ratio': .2, 'device': 'cpu'})
    for v in data['variants']:
        v['dimensions'] = [3] if v['name'] in ('ordered', 'pca') else [2, 3]
        if 'evaluation_dimensions' in v: v['evaluation_dimensions'] = [2, 3]
        if v['name'] == 'ordered': v['model_config']['prefix_dimensions'] = [1, 2, 3]
    return data


def tiny_experiment(tmp_path, phase):
    from mlx.modes.text_embedding.commands import EmbedTextCommand
    from mlx.modes.text_embedding.experiment import BenchmarkAutoencoderRetrieval
    from mlx.modes.text_embedding.experiment_requests import AutoencoderRetrievalRequest
    for name in ['one', 'two']:
        root = tmp_path / 'datasets' / name
        (root / 'qrels').mkdir(parents=True, exist_ok=True)
        (root / 'corpus.jsonl').write_text('\n'.join(json.dumps({'_id': f'd{i}', 'title': 'T', 'text': f'document {i}'}) for i in range(12)))
        (root / 'queries.jsonl').write_text(json.dumps({'_id': 'q', 'text': 'question'}))
        (root / 'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nq\td1\t1\n')
    model = tmp_path / 'model.gguf'; model.touch()
    config = tmp_path / f'{phase}.json'; config.write_text(json.dumps(tiny_config(phase)))
    calls = []
    class Provider:
        dimensions = 4
        def runtime_metadata(self):
            from importlib.metadata import version, PackageNotFoundError
            try:
                binding_version = version('llama-cpp-python')
            except PackageNotFoundError:
                binding_version = None
            return {'context_length': 2048, 'pooling_effective': 'mean',
                    'llama_cpp_python_version': binding_version, 'truncation': 'backend-token-prefix'}
        def embed(self, texts):
            calls.extend(texts)
            return [[len(t), sum(map(ord,t)) % 11, 1, sum(map(ord,t)) % 7] for t in texts]
    def embed(request, **kwargs):
        return EmbedTextCommand(request, provider_factory=lambda *a, **k: Provider(), **kwargs)
    request = AutoencoderRetrievalRequest(experiment_config=str(config), dataset_path=str(tmp_path / 'datasets'),
                                         model=str(model), output_path=str(tmp_path / phase), resume=True)
    def command(r=request):
        return BenchmarkAutoencoderRetrieval(r, training_factory=lambda options: TrainAutoencoder(AutoencoderTrainRequest(**options)),
            transformer_factory=AutoencoderRepresentationTransformer, embed_factory=embed,
            vector_loader=EmbeddingCsvLoader(), variant_validator=validate_training_variant)
    return command, request, calls


def test_configured_end_to_end_counts_resume_and_reports(tmp_path):
    command, request, calls = tiny_experiment(tmp_path, 'confirmation')
    dry = command(replace(request, dry_run=True)).execute()
    assert dry['training_runs'] == 44 and not Path(request.output_path).exists() and not calls
    result = command().execute()
    assert result['training_runs'] == 44 and len(calls) == 26
    root = Path(request.output_path)
    report = json.loads((root / 'comparison/statistics.json').read_text())
    assert len(report['primary']) == 16 and len(report['secondary']) == 8
    with (root / 'comparison/training_runs.csv').open() as f:
        runs = list(csv.DictReader(f))
    assert len(runs) == 48  # 44 neural runs + 4 PCA fits, ordered training counted once.
    command().execute()
    assert len(calls) == 26
    changed = root / 'one/runs/ordered-3/seed-42/training/autoencoder_manifest.json'
    changed.write_text('{}')
    with pytest.raises(MLXUserError, match='changed'):
        command().execute()


def test_pilot_selection_without_retrieval_and_tamper_detection(tmp_path):
    from mlx.modes.text_embedding.experiment_selection import SelectAutoencoderExperimentSettings
    command, request, calls = tiny_experiment(tmp_path, 'pilot')
    result = command().execute()
    root = Path(request.output_path)
    assert result['retrieval_evaluations'] == 0
    assert not list(root.rglob('query_metrics.csv'))
    output = tmp_path / 'selected'
    selection = SelectAutoencoderExperimentSettings(root, output).execute()
    frozen = load_experiment_config(selection['config'])
    assert frozen.counts()['training_runs'] == 550
    assert set(selection['selected']) == {'mse-least-volume', 'mse-covariance'}
    (root / 'pilot/results.json').write_text('{}')
    with pytest.raises(MLXUserError):
        SelectAutoencoderExperimentSettings(root, tmp_path / 'bad').execute()


def test_imported_source_is_verified_and_never_embedded_again(tmp_path):
    from mlx.modes.text_embedding.experiment_artifacts import file_hashes
    from mlx.core.artifacts import write_json_atomic
    command, request, calls = tiny_experiment(tmp_path, 'confirmation')
    command().execute()
    count = len(calls)
    imported = replace(request, embedding_source=request.output_path, output_path=str(tmp_path / 'reuse'), dry_run=True)
    command(imported).execute()
    assert len(calls) == count and not Path(imported.output_path).exists()
    with pytest.raises(MLXUserError, match='context_length'):
        command(replace(imported, context_length=512)).execute()
    source = Path(request.output_path) / 'one/original/corpus_embeddings.csv'
    source.write_text(source.read_text() + '\n')
    with pytest.raises(MLXUserError, match='modified'):
        command(imported).execute()


def test_selection_missing_cells_and_tie_rule(tmp_path):
    from mlx.modes.text_embedding.experiment_artifacts import file_hashes
    from mlx.modes.text_embedding.experiment_selection import SelectAutoencoderExperimentSettings
    from mlx.core.artifacts import write_json_atomic
    command, request, calls = tiny_experiment(tmp_path, 'pilot')
    command().execute()
    root = Path(request.output_path)
    results = json.loads((root / 'pilot/results.json').read_text())
    for cell in results['cells']:
        cell['reconstruction'] = 1.0
    write_json_atomic(root / 'pilot/results.json', results)
    write_json_atomic(root / 'pilot/stage.json', {'hashes': file_hashes(root / 'pilot')})
    output = SelectAutoencoderExperimentSettings(root, tmp_path / 'tie').execute()
    assert output['selected'] == {'mse-least-volume': .01, 'mse-covariance': .01}
    results['cells'].pop()
    write_json_atomic(root / 'pilot/results.json', results)
    write_json_atomic(root / 'pilot/stage.json', {'hashes': file_hashes(root / 'pilot')})
    with pytest.raises(MLXUserError, match='missing'):
        SelectAutoencoderExperimentSettings(root, tmp_path / 'missing').execute()


def test_cli_variant_conflicts_and_runtime_overrides(tmp_path):
    from mlx.cli import build_parser, _build_config
    from mlx.cli_config import explicit_option_destinations
    from mlx.modes.text_embedding.runner import run_text_embedding
    from mlx.modes.text_embedding.experiment_requests import AutoencoderRetrievalRequest
    parser = build_parser()
    args = ['--mode', 'text-embedding', '--action', 'benchmark-autoencoders', '--experiment-config', 'config.json', '--losses', 'mse']
    config = _build_config(parser.parse_args(args))
    config['_explicit_options'] = explicit_option_destinations(parser, args)
    with pytest.raises(MLXUserError, match='conflicts'):
        run_text_embedding(config)
    path = tmp_path / 'config.json'; path.write_text(json.dumps(pilot_config()))
    request = AutoencoderRetrievalRequest(epochs=2, extras={'explicit_options': ['epochs']})
    resolved = load_experiment_config(path).resolve_request(request)
    assert resolved.epochs == 2 and resolved.hidden_dim == 512 and resolved.seeds == (40, 41)


def test_real_gguf_configured_smoke(tmp_path):
    import os
    model = os.environ.get('MLX_RETRIEVAL_SMOKE_MODEL')
    if not model:
        pytest.skip('Set MLX_RETRIEVAL_SMOKE_MODEL to run the real GGUF v2 smoke test.')
    command, request, calls = tiny_experiment(tmp_path, 'confirmation')
    from mlx.modes.text_embedding.commands import EmbedTextCommand
    real = command(replace(request, model=model))
    real.embed_factory = EmbedTextCommand
    result = real.execute()
    assert Path(result['report']).is_file() and not calls


def test_regularizer_calibration_rejects_nonpositive_and_nonfinite():
    class Constant(nn.Module):
        def encode(self, x): return torch.ones(len(x), 2)
        def decode(self, z): return torch.ones(len(z), 4)
    with pytest.raises(MLXUserError, match='positive'):
        CovarianceLoss().calibrate(Constant(), torch.zeros(4, 4))
    with pytest.raises(MLXUserError, match='finite'):
        LeastVolumeLoss().calibrate(Constant(), torch.full((4, 4), float('nan')))


def test_spectral_control_and_volume_match_initialization_and_batch_order(tmp_path):
    source = tmp_path / 'vectors.csv'
    with source.open('w') as f:
        writer = csv.writer(f); writer.writerow(['embedding'])
        writer.writerows([json.dumps(row)] for row in np.random.default_rng(1).normal(size=(13, 4)).tolist())
    class Trace(TrainAutoencoder):
        def _train_epoch(self, model, loader, criterion, optimizer):
            self.weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.batches = [x.tolist() for x in loader]
            return 0.0
        def _validation_loss(self, model, loader, criterion): return 0.0
    traces = []
    for loss in ['mse', 'mse-least-volume']:
        trainer = Trace(AutoencoderTrainRequest(model='simple-spectral', loss=loss, input_path=str(source),
            output_path=str(tmp_path / loss), hidden_dim=4, bottleneck_dim=3, minimum_batch_size=2,
            batch_size=4, epochs=1, plots=False))
        trainer.execute(); traces.append(trainer)
    assert traces[0].batches == traces[1].batches
    assert all(torch.equal(v, traces[1].weights[k]) for k, v in traces[0].weights.items())


@pytest.mark.parametrize('update', [
    {'dimensions': [[2]]}, {'model': 42}, {'kind': 'pca', 'model': 'simple'},
])
def test_malformed_variants_are_user_errors(tmp_path, update):
    config = pilot_config(); config['variants'][0].update(update)
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(MLXUserError): load_experiment_config(path)
