import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.dawn_priority import RunDawnPriorityStudy


def test_priority_sequence_and_fixed_protocol(tmp_path, monkeypatch):
    events = []
    def factory(name, value):
        def construct(*args, **kwargs):
            events.append((name, args, kwargs))
            return SimpleNamespace(execute=lambda: value)
        return construct
    command = RunDawnPriorityStudy(
        {'output':str(tmp_path),'checkpoint':'foundation.pt','dataset':'dawn','export':'complete.pt'},
        calibrator=factory('calibrate', {'selected_physical_batch_size':8}),
        experiment=factory('experiment', []), reporter=factory('report', {}),
        exporter=factory('export', {'reload_exact':True}))
    monkeypatch.setattr(command, '_verify_inputs', lambda: None)
    monkeypatch.setattr(command, '_html', lambda path: None)
    monkeypatch.setattr(command, '_verify_smoke_protocol', lambda *args: None)
    command.execute()
    assert [e[0] for e in events] == ['calibrate','experiment','experiment','report','export']
    request = events[2][1][0]
    assert request.seeds == (1,2,3,4,5) and request.epochs == 20
    assert request.methods == ('drax-residual-fusion','lora')
    assert request.train_head is False and request.head_policy == 'preserve'
    assert request.seed_adapter_initialization and request.reduction == 16
    assert request.batch_size == 8 and request.device == 'cuda'
    assert json.loads((tmp_path/'status.json').read_text())['status'] == 'completed'


def test_calibration_does_not_silently_reduce_batch(tmp_path, monkeypatch):
    command = RunDawnPriorityStudy(
        {'output':str(tmp_path),'checkpoint':'foundation.pt','dataset':'dawn','export':'complete.pt'},
        calibrator=lambda *a,**k:SimpleNamespace(execute=lambda:{'selected_physical_batch_size':4}),
        experiment=lambda *a,**k:pytest.fail('Training must not start'))
    monkeypatch.setattr(command, '_verify_inputs', lambda: None)
    with pytest.raises(MLXUserError, match='protocol was not changed'):
        command.execute()
    assert json.loads((tmp_path/'status.json').read_text())['status'] == 'failed'


def test_rank100_addon_uses_matched_protocol_and_separate_artifacts(tmp_path, monkeypatch):
    from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
    calls = []
    def experiment(request):
        calls.append(request)
        return SimpleNamespace(execute=lambda:[{'method':'lora','seed':s} for s in request.seeds])
    calibration = []
    def calibrator(*args, **kwargs):
        calibration.append(kwargs)
        return SimpleNamespace(execute=lambda:{'selected_physical_batch_size':8})
    command = RunDawnPriorityStudy({'output':str(tmp_path)},experiment=experiment,calibrator=calibrator)
    monkeypatch.setattr(command,'_verify_smoke_protocol',lambda *a,**k:None)
    request = AdapterExperimentRequest('yolox-l',Path('foundation'),Path('dawn'),tmp_path/'study',
                                       ('lora',),(1,2,3,4,5),epochs=20,batch_size=8,
                                       seed_adapter_initialization=True)
    rows=command._lora100(request)
    assert calibration[0]['rank']==100 and calibration[0]['alpha']==12.5
    assert calls[0].epochs==1 and calls[1].epochs==20
    assert calls[1].seeds==(1,2,3,4,5) and calls[1].rank==100
    assert calls[1].alpha==12.5 and not calls[1].train_head
    assert calls[1].output==tmp_path/'lora-r100-study'
    assert len(rows)==5 and all(r['method']=='lora-r100' and r['source_run'] for r in rows)


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
