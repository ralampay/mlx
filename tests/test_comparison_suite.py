import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.comparison_data import PrepareComparisonDataset
from mlx.modes.object_detection.comparison_suite import balanced_schedule, admit_datasets, RunComparisonSuite, include_snapshot_file
from mlx.modes.object_detection.comparison_report import AnalyzeComparisonSuite
from mlx.modes.object_detection.libreyolo.comparison_backend import LibreYOLOComparisonBackend, validate_detection_outputs


def test_schedule_keeps_complete_seed_blocks_and_cycles_permutations():
    schedule = balanced_schedule(('a','b','c'),range(6))
    assert len(schedule)==18
    assert len({tuple(model for seed,model in schedule if seed==s) for s in range(6)})==6


def test_snapshot_preserves_package_metadata_and_notices_without_environment(tmp_path):
    assert include_snapshot_file(tmp_path/'package.egg-info/PKG-INFO',tmp_path)
    assert include_snapshot_file(tmp_path/'LICENSE',tmp_path)
    assert include_snapshot_file(tmp_path/'NOTICE',tmp_path)
    assert not include_snapshot_file(tmp_path/'env/package/metadata.json',tmp_path)


def test_budget_selects_priority_prefix_with_contingency():
    assert admit_datasets([('a',10),('b',20),('c',5)],40)==(['a','b'],37.5)
    with pytest.raises(MLXUserError,match='budget'):
        admit_datasets([('a',10)],5)


def test_dataset_deduplication_preserves_test_and_originals(tmp_path):
    source=tmp_path/'source'
    rng=np.random.default_rng(3)
    duplicate=rng.integers(0,256,(20,20,3),dtype=np.uint8)
    for split in ('test','val','train'):
        for kind in ('images','labels'):
            (source/kind/split).mkdir(parents=True)
        pixels=duplicate if split in ('test','train') else rng.integers(0,256,(20,20,3),dtype=np.uint8)
        Image.fromarray(pixels).save(source/'images'/split/'a.png')
        (source/'labels'/split/'a.txt').write_text('0 .5 .5 .4 .4\n')
        Image.fromarray(rng.integers(0,256,(20,20,3),dtype=np.uint8)).save(source/'images'/split/'b.png')
        (source/'labels'/split/'b.txt').write_text('0 .5 .5 .4 .4\n')
    (source/'data.yaml').write_text(yaml.safe_dump(dict(names=['object'],**{s:f'images/{s}' for s in ('test','val','train')})))
    result=PrepareComparisonDataset(source,tmp_path/'clean').execute()
    assert len(result['excluded'])==1
    assert result['excluded'][0]['split']=='train'
    assert (source/'images/train/a.png').exists()
    assert not (tmp_path/'clean/images/train/a.png').exists()
    with pytest.raises(MLXUserError,match='already exists'):
        PrepareComparisonDataset(source,tmp_path/'clean').execute()


def test_incomplete_report_emits_no_inferential_claims(tmp_path):
    protocol=dict(datasets=['d'],models=['a','b'],seeds=[1,2],equivalence_margin_pp=1.,alpha=.05)
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    result=AnalyzeComparisonSuite(tmp_path).execute()
    assert result['status']=='incomplete'
    assert result['claims']==[]
    assert (tmp_path/'aggregate/summary.html').exists()


def test_completed_report_corrects_all_claims_together(tmp_path):
    protocol=dict(datasets=['d'],models=['a','b','c'],seeds=[1,2,3],equivalence_margin_pp=1.,alpha=.05)
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    for i,model in enumerate(protocol['models']):
        for seed in protocol['seeds']:
            directory=tmp_path/'runs/d'/model/f'seed-{seed}'
            directory.mkdir(parents=True)
            record=dict(dataset='d',model=model,seed=seed,training_seconds=5,
                        metrics=dict(map_50_95=.3+i*.01+seed*i*.001,map_50=.5))
            (directory/'complete.json').write_text(json.dumps(record))
    result=AnalyzeComparisonSuite(tmp_path).execute()
    assert len(result['claims'])==6
    assert all(c['holm_p']>=c['p'] for c in result['claims'])
    assert (tmp_path/'aggregate/results.csv').exists()


def test_prepare_stops_on_unresolved_audit_before_creating_output(tmp_path):
    dataset=tmp_path/'dataset'
    dataset.mkdir()
    (dataset/'audit.json').write_text('{"status":"needs-review"}')
    class Backend:
        def verify_device(self): return {}
    config=dict(output=str(tmp_path/'run'),models=['a','b'],seeds=[1,2],epochs=2,image_size=32,budget_hours=1,datasets={'d':str(dataset)})
    with pytest.raises(MLXUserError,match='unresolved'):
        RunComparisonSuite(config,Backend(),None).execute('prepare')
    assert not (tmp_path/'run').exists()


def test_cuda_required_failure(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(MLXUserError,match='Training was not started'):
        LibreYOLOComparisonBackend().verify_device()


def test_reload_validation_accepts_native_pyramid_and_decoded_outputs():
    import torch
    validate_detection_outputs([torch.zeros(1,7,8,8),torch.zeros(1,7,4,4)],2)
    validate_detection_outputs(torch.zeros(1,10,7),2)
    for outputs in ([],torch.zeros(1,10,6),[torch.full((1,7,8,8),float('nan'))]):
        with pytest.raises(MLXUserError,match='shape/finite'):
            validate_detection_outputs(outputs,2)


def test_interrupted_directory_is_never_silently_reused(tmp_path):
    root=tmp_path/'study'
    directory=root/'runs/d/a/seed-1'
    directory.mkdir(parents=True)
    (root/'manifest.json').write_text('{}')
    command=RunComparisonSuite({'output':str(root)},None,None)
    with pytest.raises(MLXUserError,match='Interrupted run preserved'):
        command._condition('d','a',1,8,False)


@pytest.mark.parametrize('capacity',[None,120,0,-1,True])
def test_provider_annotation_capacity_contract(tmp_path,capacity):
    from mlx.modes.object_detection.libreyolo.training import TrainLibreYOLOObjectDetection
    command=TrainLibreYOLOObjectDetection({'max_labels':capacity})
    kwargs=dict(data='data.yaml',project_dir=tmp_path,run_name='run',resume=False,allow_pretrained=True)
    if capacity is not None and (isinstance(capacity,bool) or capacity < 1):
        with pytest.raises(MLXUserError,match='max_labels'):
            command._build_train_kwargs(**kwargs)
    else:
        result=command._build_train_kwargs(**kwargs)
        assert result.get('max_labels')==capacity
