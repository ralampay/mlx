from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
from mlx.modes.object_detection.neu_det import grouped_split
from mlx.modes.object_detection.adapter_transfer import RunQueuedTransferStudy, active_cuda_jobs


def test_grouped_stratification_reproducible_and_no_duplicate_leakage():
    records = [{"stem":f"{c}_{i}","source_class":c,"pixel_sha256":f"{c}_{i}"}
               for c in ("a","b") for i in range(300)]
    first = grouped_split(records)
    assert first == grouped_split(list(reversed(records)))
    assert Counter(first.values()) == {"train":420,"val":90,"test":90}
    records.append({**records[0],"stem":"duplicate"})
    assignments = grouped_split(records)
    assert assignments["duplicate"] == assignments[records[0]["stem"]]


def test_transfer_request_requires_head_and_forbids_frozen(tmp_path):
    request = AdapterExperimentRequest("yolox-l",tmp_path,tmp_path,tmp_path,("lora",),(1,),head_policy="reset-classifiers")
    with pytest.raises(MLXUserError,match="train-head"):
        request.validate()
    replace(request,train_head=True).validate()
    with pytest.raises(MLXUserError):
        replace(request,train_head=True,methods=("frozen",)).validate()
    parsed = build_parser().parse_args(["--head-policy","reset-classifiers","--train-head"])
    assert parsed.head_policy == "reset-classifiers" and parsed.train_head


def test_queue_detects_compute_and_ignores_desktop(monkeypatch):
    monkeypatch.setattr("subprocess.run",lambda *a,**k:SimpleNamespace(stdout="12, /usr/bin/ptyxis\n34, /env/bin/python\n"))
    assert active_cuda_jobs() == [{"pid":34,"name":"/env/bin/python"}]


def test_queue_waits_then_runs_without_overwrite(tmp_path,monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"output":str(tmp_path)}))
    command = RunQueuedTransferStudy(path,poll_seconds=0)
    calls = []
    monkeypatch.setattr(command,"_verify",lambda config:None)
    monkeypatch.setattr(command,"_run",lambda config:calls.append("run"))
    statuses = iter([[{"pid":123}],[]])
    monkeypatch.setattr("mlx.modes.object_detection.adapter_transfer.active_cuda_jobs",lambda:next(statuses))
    command.execute()
    assert calls == ["run"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "queued"
    with pytest.raises(MLXUserError,match="resume"):
        command.execute()


def test_queue_records_failure(tmp_path,monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"output":str(tmp_path)}))
    command = RunQueuedTransferStudy(path)
    def fail(config):
        raise MLXUserError("checksum mismatch")
    monkeypatch.setattr(command,"_verify",fail)
    with pytest.raises(MLXUserError,match="checksum"):
        command.execute()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"


def test_transfer_dataset_rejects_yaml_taxonomy_mismatch(tmp_path):
    from mlx.modes.object_detection.adapter_data import load_prepared_adapter_dataset
    (tmp_path / "manifest.json").write_text(json.dumps({"classes":["defect"]}))
    (tmp_path / "data.yaml").write_text("names: [car]\n")
    with pytest.raises(MLXUserError,match="YAML class order"):
        load_prepared_adapter_dataset(tmp_path,expected_classes=None)


def test_calibration_requires_cuda(tmp_path):
    from mlx.modes.object_detection.libreyolo.transfer_backend import CalibrateTransferBatch
    request = AdapterExperimentRequest("yolox-l",tmp_path,tmp_path,tmp_path,("lora",),(1,),device="cpu")
    with pytest.raises(MLXUserError,match="requires CUDA"):
        CalibrateTransferBatch(request,["defect"]).execute()


def test_epoch_measurements_match_one_based_events():
    import torch
    from mlx.modes.object_detection.libreyolo.adapter_backend import _AdapterYOLOXTrainerMixin
    class Base:
        def _train_epoch(self,epoch):
            return "result"
    class Trainer(_AdapterYOLOXTrainerMixin,Base):
        pass
    trainer = Trainer()
    trainer.device = torch.device("cpu")
    trainer.wrapper_model = SimpleNamespace(_adapter_measure_epochs=True,_adapter_epoch_cuda=[])
    assert trainer._train_epoch(0) == "result"
    assert trainer.wrapper_model._adapter_epoch_cuda[0]["epoch"] == 1
    assert trainer.wrapper_model._adapter_epoch_cuda[0]["peak_cuda_memory_mb"] is None
