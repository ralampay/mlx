import json

import pytest

from mlx.core.exceptions import MLXUserError


def test_report_includes_aliased_runs_and_rejects_duplicates(tmp_path):
    from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
    extra=[]
    for method,ap in [('drax-residual-fusion',.4),('custom-lora',.42)]:
        for seed in range(1,3):
            extra.append({'method':method,'seed':seed,'status':'completed','mAP50':ap,
                          'mAP50_95':ap,'precision':ap,'recall':ap,'trainable_percent':1.,
                          'training_seconds':1.,'source_run':str(tmp_path / method / str(seed))})
    GenerateAdapterReport(tmp_path,comparison_method='drax-residual-fusion',extra_runs=tuple(extra)).execute()
    data=json.loads((tmp_path/'aggregate/results.json').read_text())
    pair=data['paired_differences']['drax-residual-fusion_minus_custom-lora']
    assert pair['n']==2 and pair['mean']==pytest.approx(-.02)
    with pytest.raises(MLXUserError,match='Duplicate'):
        GenerateAdapterReport(tmp_path,extra_runs=tuple(extra+[extra[0]])).execute()


