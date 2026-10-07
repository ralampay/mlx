import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError


def test_report_includes_rank100_paired_differences_and_rejects_duplicates(tmp_path):
    from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
    extra=[]
    for method,ap in [('drax-residual-fusion',.4),('lora-r100',.42)]:
        for seed in range(1,6):
            extra.append({'method':method,'seed':seed,'status':'completed','mAP50':ap,
                          'mAP50_95':ap,'precision':ap,'recall':ap,'trainable_percent':1.,
                          'training_seconds':1.,'source_run':f'/study/{method}/{seed}'})
    GenerateAdapterReport(tmp_path,comparison_method='drax-residual-fusion',extra_runs=tuple(extra)).execute()
    data=json.loads((tmp_path/'aggregate/results.json').read_text())
    pair=data['paired_differences']['drax-residual-fusion_minus_lora-r100']
    assert pair['n']==5 and pair['mean']==pytest.approx(-.02)
    with pytest.raises(MLXUserError,match='Duplicate'):
        GenerateAdapterReport(tmp_path,extra_runs=tuple(extra+[extra[0]])).execute()


@pytest.mark.parametrize('exit_code', [0, 1])
def test_supervisor_restores_held_queues_after_child_exit(tmp_path, monkeypatch, exit_code):
    path = Path(__file__).parents[1]/'scripts/experiments/dawn_adapter_priority.py'
    spec = importlib.util.spec_from_file_location('priority_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path/'plan.json'; config.write_text(json.dumps({'output':str(tmp_path)}))
    state = tmp_path/'previous.json'; state.write_text(json.dumps({'status':'completed'}))
    held = tmp_path/'held.json'; held.write_text(json.dumps({'processes':[{'pid':123}], 'active_condition_status':str(state)}))
    monkeypatch.setattr(module, 'matching_process', lambda p:True)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a,**k:SimpleNamespace(wait=lambda:exit_code,poll=lambda:exit_code))
    from mlx.modes.object_detection import adapter_queue
    monkeypatch.setattr(adapter_queue, 'active_cuda_jobs', lambda:[])
    signals = []
    monkeypatch.setattr(module.os, 'kill', lambda pid,sig:signals.append((pid,sig)))
    command = module.SupervisePriority(config, held)
    if exit_code:
        with pytest.raises(MLXUserError, match='exited'):
            command.execute()
    else:
        assert command.execute() == 0
    assert signals == [(123,module.signal.SIGCONT)]
    assert json.loads((tmp_path/'queue-restoration.json').read_text())[0]['resumed']


def test_suspended_cuda_client_is_allowed_only_when_verified_stopped(tmp_path, monkeypatch):
    path = Path(__file__).parents[1]/'scripts/experiments/dawn_adapter_priority.py'
    spec = importlib.util.spec_from_file_location('priority_suspended', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config=tmp_path/'plan.json';config.write_text(json.dumps({'output':str(tmp_path)}))
    held=tmp_path/'held.json';held.write_text(json.dumps({'mode':'suspended-in-memory','processes':[{'pid':123}]}))
    monkeypatch.setattr(module,'matching_process',lambda p:True)
    monkeypatch.setattr(module,'stopped_process',lambda p:True)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:SimpleNamespace(wait=lambda:0,poll=lambda:0))
    from mlx.modes.object_detection import adapter_queue
    monkeypatch.setattr(adapter_queue,'active_cuda_jobs',lambda:[{'pid':123}])
    signals=[]
    monkeypatch.setattr(module.os,'kill',lambda *args:signals.append(args))
    assert module.SupervisePriority(config,held).execute()==0
    assert signals==[(123,module.signal.SIGCONT)]
    monkeypatch.setattr(module,'stopped_process',lambda p:False)
    with pytest.raises(MLXUserError,match='not stopped'):
        module.SupervisePriority(config,held).execute()
