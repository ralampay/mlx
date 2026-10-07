from dataclasses import replace
import json

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
from mlx.modes.object_detection.adapter_transfer import RunQueuedTransferStudy
from mlx.modes.object_detection.transfer_report import GenerateTaxonomyTransferReport


def test_condition_identity_does_not_change_backend(tmp_path):
    request = AdapterExperimentRequest("yolox-l",tmp_path,tmp_path,tmp_path,("lora",),(1,),
        head_policy="reset-classifiers",train_head=True,rank=100,alpha=12.5,condition_id="lora-r100")
    request.validate()
    assert request.methods == ("lora",)
    assert request.method_id("lora") == "lora-r100"
    assert request.alpha/request.rank == .125
    with pytest.raises(MLXUserError):
        replace(request,condition_id="../lora").validate()
    with pytest.raises(MLXUserError):
        replace(request,methods=("lora","ssf")).validate()


def test_declare_two_condition_study(tmp_path):
    config = {"checkpoint":"foundation.pt","checkpoint_sha256":"hash","dataset_selection_sha256":"split"}
    conditions = [{"id":"lora-r100"},{"id":"drax-spatial"}]
    RunQueuedTransferStudy._declare_study(config,tmp_path,conditions,[1,2,3,4,5])
    study = json.loads((tmp_path / "study.json").read_text())
    assert study["available_methods"] == ["lora-r100","drax-spatial"]
    assert len(study["planned_seeds"])*len(study["available_methods"]) == 10


def test_combined_report_has_four_holm_comparisons(tmp_path):
    rows = []
    for method in ("head-only","lora","ssf","convpass","drax-hybrid","full-finetune","lora-r100","drax-spatial"):
        for seed in range(1,6):
            rows.append({"method":method,"seed":seed,"mAP50":.5,"mAP50_95":.3+seed*.001,
                "precision":.6,"recall":.7,"training_seconds":100.,"peak_cuda_memory_mb":1000.,
                "end_to_end_inference_ms":20.,"trainable_params":9000000})
    pairs = GenerateTaxonomyTransferReport._combined_report(rows,tmp_path)
    assert len(pairs) == 4
    assert all(row["holm_adjusted_p"] == 1 for row in pairs.values())
    assert len(json.loads((tmp_path / "results.json").read_text())["runs"]) == 40
    assert "cached-prediction COCO" in (tmp_path / "summary.md").read_text()


def test_baseline_rejects_changed_dataset_before_reuse(tmp_path):
    from mlx.modes.object_detection.transfer_baseline import VerifyTransferBaseline
    (tmp_path / "status.json").write_text(json.dumps({"status":"completed","completed_runs":30}))
    (tmp_path / "plan.json").write_text(json.dumps({"dataset_selection_sha256":"original"}))
    with pytest.raises(MLXUserError,match="dataset_selection_sha256"):
        VerifyTransferBaseline(tmp_path,{"dataset_selection_sha256":"changed"}).execute()
