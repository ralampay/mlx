import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.dawn_paired import RunPairedDawnStudy
from mlx.modes.object_detection.dawn_paired_report import GeneratePairedDawnReport


def configuration(root):
    return {'output':str(root),'checkpoint':'foundation.pt','dataset':'dataset',
            'seeds':list(range(201,221)),
            'analysis':{'metric':'mAP50_95','equivalence_margin':.01,'family_alpha':.05,'multiplicity':'holm-six-claims'},
            'conditions':[{'id':label,'method':method,'rank':rank,'alpha':alpha}
                          for label,method,rank,alpha in [
                              ('drax-residual-fusion','drax-residual-fusion',8,1.),
                              ('lora-r4','lora',4,.5),('lora-r100','lora',100,12.5),
                              ('full-finetune','full-finetune',8,1.)]]}


def test_fixed_requests_and_sequence(tmp_path,monkeypatch):
    events=[]
    def factory(name,result):
        def construct(*args,**kwargs):
            events.append((name,args,kwargs))
            return SimpleNamespace(execute=lambda:result)
        return construct
    command=RunPairedDawnStudy(configuration(tmp_path),
        experiment=factory('experiment',[]),
        calibrator=factory('calibrate',{'selected_physical_batch_size':8}),
        reporter=factory('report',{}))
    monkeypatch.setattr(command,'_verify',lambda:None)
    monkeypatch.setattr(command,'_check_protocol',lambda requests:None)
    monkeypatch.setattr(command,'_release_cache',lambda:None)
    command.execute()
    assert [e[0] for e in events[:8]]==['calibrate']*4+['experiment']*4
    full=[e[1][0] for e in events if e[0]=='experiment' and e[1][0].epochs==20]
    assert len(full)==4
    assert all(r.seeds==tuple(range(201,221)) and r.batch_size==8 and r.device=='cuda'
               and r.seed_adapter_initialization and not r.train_head for r in full)
    assert (full[1].rank,full[1].alpha)==(4,.5)
    assert (full[2].rank,full[2].alpha)==(100,12.5)
    assert events[-1][2]['final'] is True
    assert json.loads((tmp_path/'status.json').read_text())['trained_runs']==80


def test_calibration_failure_stops_without_changing_recipe(tmp_path,monkeypatch):
    command=RunPairedDawnStudy(configuration(tmp_path),
        calibrator=lambda *a,**k:SimpleNamespace(execute=lambda:{'selected_physical_batch_size':4}),
        experiment=lambda *a,**k:pytest.fail('must not train'))
    monkeypatch.setattr(command,'_verify',lambda:None)
    with pytest.raises(MLXUserError,match='protocol not changed'):
        command.execute()
    assert json.loads((tmp_path/'status.json').read_text())['status']=='failed'


def test_no_restarting_over_existing_state(tmp_path):
    (tmp_path/'status.json').write_text('{}')
    with pytest.raises(MLXUserError,match='already has state'):
        RunPairedDawnStudy(configuration(tmp_path)).execute()


def test_duplicate_seeds_and_changed_recipe_rejected(tmp_path):
    config=configuration(tmp_path)
    config['seeds']=[201]*20
    with pytest.raises(MLXUserError,match='distinct paired seeds'):
        RunPairedDawnStudy(config)._verify()
    config=configuration(tmp_path)
    config['conditions'][1]['rank']=8
    with pytest.raises(MLXUserError,match='declared recipe'):
        RunPairedDawnStudy(config)._verify()


def test_incomplete_final_report_rejected(tmp_path):
    config=configuration(tmp_path)
    path=tmp_path/'runs/lora-r4/lora/seed-201/metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seed':201,'method':'lora','status':'completed','mAP50_95':.4}))
    with pytest.raises(MLXUserError,match='every prespecified seed'):
        GeneratePairedDawnReport(config,final=True).execute()


def test_final_report_six_adjusted_claims(tmp_path,monkeypatch):
    from mlx.modes.object_detection import dawn_paired_report as module
    config=configuration(tmp_path)
    for ci,condition in enumerate(config['conditions']):
        for seed in config['seeds']:
            p=tmp_path/'runs'/condition['id']/condition['method']/f'seed-{seed}'/'metrics.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'seed':seed,'method':condition['method'],'status':'completed',
                                    'mAP50_95':.4-ci*.003+(seed%3)*.0001*(ci+1)}))
    def generic(*a,**k):
        (tmp_path/'aggregate').mkdir(exist_ok=True)
        return SimpleNamespace(execute=lambda:None)
    monkeypatch.setattr(module,'GenerateAdapterReport',generic)
    monkeypatch.setattr(module,'AnalyzePairedDifferences',lambda d,**k:SimpleNamespace(execute=lambda:SimpleNamespace(to_dict=lambda:{
        'standard_deviation':.001,'mean_difference':sum(d)/len(d),'tost_lower_p':.001,'tost_upper_p':.002})))
    result=GeneratePairedDawnReport(config,final=True).execute()
    assert result['run_count']==80 and len(result['claims'])==6
    assert all(c['p_holm']>=c['p_raw'] for c in result['claims'])
    assert (tmp_path/'aggregate/statistical-report.html').exists()
