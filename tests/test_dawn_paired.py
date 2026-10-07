import json

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


def test_no_restarting_over_existing_state(tmp_path):
    (tmp_path/'status.json').write_text('{}')
    with pytest.raises(MLXUserError,match='already has state'):
        RunPairedDawnStudy(configuration(tmp_path)).execute()


def test_incomplete_final_report_rejected(tmp_path):
    config=configuration(tmp_path)
    path=tmp_path/'runs/lora-r4/lora/seed-201/metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seed':201,'method':'lora','status':'completed','mAP50_95':.4}))
    with pytest.raises(MLXUserError,match='every prespecified seed'):
        GeneratePairedDawnReport(config,final=True).execute()
