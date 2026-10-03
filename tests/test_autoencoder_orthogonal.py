import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mlx.core.exceptions import MLXUserError
from mlx.core.partitions import partition_rows
from mlx.modes.autoencoder.architectures.orthogonal import OrthogonalTiedAutoencoder
from mlx.modes.autoencoder.cosine_loss import MSECosineLoss
from mlx.modes.autoencoder.commands import TrainAutoencoder
from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
from mlx.modes.text_embedding.compression_controls import FitLinearProjection, LinearProjectionVectors
from mlx.modes.text_embedding.experiment_config import load_experiment_config
from mlx.modes.text_embedding.experiment_templates import orthogonal_config


def test_tied_geometry_initialization_and_projection():
    torch.manual_seed(8)
    values = torch.randn(30, 6)
    model = OrthogonalTiedAutoencoder(6, 3)
    model.initialize_from_training_values(values)
    assert len(list(model.parameters())) == 1
    _, _, vh = torch.linalg.svd(values.double(), full_matrices=False)
    assert torch.allclose(model.weight.T @ model.weight, (vh[:3].T @ vh[:3]).float(), atol=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(4):
        optimizer.zero_grad()
        MSECosineLoss()(model(values), values).backward()
        assert torch.isfinite(model.weight.grad).all()
        optimizer.step()
        model.project_parameters_()
        assert torch.allclose(model.weight @ model.weight.T, torch.eye(3), atol=1e-6)
    z = model.encode(values)
    decoded = model.decode(z)
    assert torch.allclose(F.cosine_similarity(z[:-1], z[1:]), F.cosine_similarity(decoded[:-1], decoded[1:]), atol=1e-6)
    with pytest.raises(MLXUserError, match='training rows'):
        model.initialize_from_training_values(values[:2])


def test_cosine_loss_formula_singleton_and_zero():
    x = torch.tensor([[1., 0., 0.]])
    y = torch.tensor([[0., 1., 0.]], requires_grad=True)
    loss, components = MSECosineLoss().components(y, x)
    assert float(loss.detach()) == pytest.approx(1.)
    assert float(components['weighted_cosine'].detach()) == pytest.approx(1/3)
    loss.backward()
    assert torch.isfinite(y.grad).all()
    assert torch.equal(MSECosineLoss(0)(y, x), F.mse_loss(y, x))
    zeros = torch.zeros(1, 3, requires_grad=True)
    MSECosineLoss()(zeros, x).backward()
    assert torch.isfinite(zeros.grad).all()
    for weight in (-1, float('nan'), float('inf'), True):
        with pytest.raises(MLXUserError):
            MSECosineLoss(weight)


@pytest.mark.parametrize('loss', ['mse', 'mse-cosine'])
def test_training_initialization_is_training_only_and_epoch_zero(tmp_path, loss):
    values = np.random.default_rng(3).normal(size=(20, 6)).astype('float32')
    train, val = partition_rows(20, .2, 42)
    values[val] += 100  # Held-out data must not enter the fitted SVD.
    source = tmp_path / 'input.csv'
    with source.open('w') as f:
        w = csv.writer(f); w.writerow(['id','embedding'])
        w.writerows((i,json.dumps(v.tolist())) for i,v in enumerate(values))
    captured = []
    class NoUpdate(TrainAutoencoder):
        def _train_epoch(self, model, loader, criterion, optimizer):
            captured.append(model.weight.detach().clone())
            return self._validation_loss(model, loader, criterion)
    result = NoUpdate(AutoencoderTrainRequest(model='orthogonal-tied', loss=loss,
        input_path=str(source), output_path=str(tmp_path/'training'), hidden_dim=1,
        bottleneck_dim=3, epochs=1, batch_size=4, plots=False)).execute()
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint['best_epoch'] == 0
    _,_,vh = np.linalg.svd(values[sorted(train)].astype('float64'),full_matrices=False)
    assert np.allclose(captured[0].T @ captured[0], vh[:3].T @ vh[:3], atol=1e-6)
    assert checkpoint['initialization']['training_rows'] == 16
    adapter = AutoencoderRepresentationTransformer(result.checkpoint_path)
    assert np.allclose(adapter.transform(values[:2].tolist()), values[:2] @ captured[0].numpy().T)
    assert 'orthogonality_error' in checkpoint['validation_components']


def test_svd_control_centering_prefix_and_recipe(tmp_path):
    values = np.random.default_rng(4).normal(size=(20,6)) + 10
    rows, val = partition_rows(20,.2,42)
    for centered, kind in [(True,'pca'),(False,'svd')]:
        FitLinearProjection(values,rows,4,tmp_path/kind,split_hash='x',centered=centered).execute()
        adapter=LinearProjectionVectors(tmp_path/kind/f'{kind}.npz',3,kind=kind)
        assert np.allclose(adapter.mean,values[rows].mean(0) if centered else 0)
        changed=values.copy();changed[val]+=1000
        FitLinearProjection(changed,rows,4,tmp_path/(kind+'2'),split_hash='x',centered=centered).execute()
        other=LinearProjectionVectors(tmp_path/(kind+'2')/f'{kind}.npz',3,kind=kind)
        assert np.allclose(adapter.transform(values),other.transform(values))
        assert adapter.provenance['type']==kind
    path=tmp_path/'recipe.json';path.write_text(json.dumps(orthogonal_config()))
    c=load_experiment_config(path)
    assert c.counts()=={'datasets':10,'training_runs':200,'pca_fits':50,'svd_fits':50,'retrieval_evaluations':430}
    assert len(c.secondary)==5
    assert c.selection['interpretation'].startswith('exploratory')


def test_background_lock_logs_and_stale_pid(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/experiment-job.py'
    output=tmp_path/'output with spaces'
    command=[sys.executable,str(script),'start','--output',str(output),'--',sys.executable,'-c',
             'import time; print("started",flush=True); time.sleep(10)']
    first=subprocess.run(command,capture_output=True,text=True,check=True)
    info=json.loads(first.stdout)
    try:
        assert Path(info['log']).is_file()
        duplicate=subprocess.run(command,capture_output=True,text=True)
        assert duplicate.returncode==73
        assert 'startup' in duplicate.stderr
        status=json.loads(Path(info['status']).read_text())
        os.kill(status['pid'],signal.SIGTERM)
        for _ in range(100):
            status=json.loads(Path(info['status']).read_text())
            if 'exit_code' in status:break
            time.sleep(.05)
        assert status['exit_code'] != 0
        # The PID file remains, but the released lock permits a new job.
        done=subprocess.run([sys.executable,str(script),'run','--output',str(output),'--',
                             sys.executable,'-c','raise SystemExit(7)'],capture_output=True)
        assert done.returncode==7
    finally:
        try:os.kill(info['pid'],signal.SIGTERM)
        except ProcessLookupError:pass


def test_configured_orthogonal_end_to_end_and_resume(tmp_path):
    from test_autoencoder_v2 import tiny_experiment
    command, request, calls = tiny_experiment(tmp_path, 'confirmation')
    config = orthogonal_config()
    config.update(datasets=['one','two'], seeds=[42,43], training={
        'epochs':1,'batch_size':4,'hidden_dim':4,'lr':.001,'val_ratio':.2,'device':'cpu'})
    for variant in config['variants']:
        variant['dimensions'] = [3] if variant.get('kind') in ('pca','svd') else [2,3]
        if 'evaluation_dimensions' in variant:
            variant['evaluation_dimensions']=[2,3]
    Path(request.experiment_config).write_text(json.dumps(config))
    result=command().execute()
    assert result['training_runs']==16 and result['svd_fits']==4
    assert result['retrieval_evaluations']==38
    root=Path(request.output_path)
    report=json.loads((root/'comparison/statistics.json').read_text())
    assert len(report['primary'])==10 and len(report['secondary'])==10
    assert len(report['difference_tests'])==10
    assert report['selection']['interpretation'].startswith('exploratory')
    assert len(list(root.glob('*/runs/svd-*/*/training/svd.npz')))==4
    command().execute()
    assert len(calls)==26


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_cosine_low_precision_zero_is_finite(dtype):
    x=torch.zeros(1,4,dtype=dtype,requires_grad=True)
    loss=MSECosineLoss()(x,torch.zeros_like(x))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_two_sided_report_matches_scipy(tmp_path):
    from scipy import stats
    from mlx.modes.text_embedding.configured_statistics import AnalyzeConfiguredAutoencoders
    from mlx.modes.text_embedding.experiment_requests import AutoencoderRetrievalRequest
    path=tmp_path/'config.json';path.write_text(json.dumps(orthogonal_config()))
    config=load_experiment_config(path)
    means={(d,v.name,width):(.002*i-.009) for i,d in enumerate(config.datasets)
           for v in config.variants for width in (384,512)}
    report=AnalyzeConfiguredAutoencoders(AutoencoderRetrievalRequest(),config,tmp_path,[],tmp_path)._compare(means)
    expected=stats.ttest_1samp([.002*i-.009 for i in range(10)],0).pvalue
    assert report['difference_tests'][0]['p_value']==pytest.approx(expected)
    assert len(report['primary'])==10


def test_child_retains_output_lock_after_supervisor_kill(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/experiment-job.py'
    output=tmp_path/'job'
    command=[sys.executable,str(script),'start','--output',str(output),'--',sys.executable,'-c','import time; time.sleep(10)']
    info=json.loads(subprocess.run(command,capture_output=True,text=True,check=True).stdout)
    state=json.loads(Path(info['status']).read_text())
    try:
        os.kill(state['pid'],signal.SIGKILL)
        duplicate=subprocess.run(command,capture_output=True,text=True)
        assert duplicate.returncode==73
    finally:
        try:os.kill(state['child_pid'],signal.SIGTERM)
        except ProcessLookupError:pass
