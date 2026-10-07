import json
from dataclasses import replace

import pytest

from mlx.core.artifacts import sha256_file
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
from mlx.modes.object_detection.adapter_transfer import RunQueuedTransferStudy
from mlx.modes.object_detection.remote_sensing_pilot import (
    PrepareRemoteAdapterDataset, RunRemoteAdapterPilot, pilot_conditions,
)


def prepared_source(root):
    members = []
    for split in ("train", "val", "test"):
        image = root / "images" / split / "sample.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"test-image")
        label = root / "labels" / split / "sample.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 .5 .5 .1 .1\n" * (67 if split == "train" else 0))
        members.append({"split": split, "image": image.name, "sha256": sha256_file(image),
                        "pixel_sha256": "pixels", "scene": "scene-1"})
        coco = {"images": [{"id": 1, "file_name": image.name, "width": 100, "height": 100}],
                "categories": [{"id": 0, "name": "ship"}],
                "annotations": [{"id": i, "image_id": 1, "category_id": 0,
                                 "bbox": [45, 45, 10, 10], "area": 100, "iscrowd": 0}
                                for i in range(67 if split == "train" else 0)]}
        (root / f"instances_{split}.json").write_text(json.dumps(coco))
    (root / "manifest.json").write_text(json.dumps({"dataset": "SSDD", "class_names": ["ship"],
        "members": members, "splits": {s: {"images": 1} for s in ("train", "val", "test")},
        "split_policy": "preserved"}))


def test_remote_view_preserves_dense_and_negative_images(tmp_path):
    source, target = tmp_path / "source", tmp_path / "view"
    prepared_source(source)
    command = PrepareRemoteAdapterDataset(source, target)
    manifest = command.execute()
    assert manifest["max_labels"] == 67
    assert len(manifest["selected_images"][0]["boxes"]) == 67
    assert manifest["selected_images"][1]["boxes"] == []
    assert (target / "images/train/sample.jpg").is_symlink()
    assert json.loads((target / "train.json").read_text()) == json.loads((source / "instances_train.json").read_text())
    assert command.execute() == manifest
    (source / "instances_train.json").write_text("{}")
    with pytest.raises(MLXUserError, match="file changed"):
        command.execute()


def test_remote_view_rejects_class_mismatch_before_writes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "view"
    prepared_source(source)
    path = source / "instances_train.json"
    content = json.loads(path.read_text())
    content["categories"][0]["name"] = "aircraft"
    path.write_text(json.dumps(content))
    with pytest.raises(MLXUserError, match="class order"):
        PrepareRemoteAdapterDataset(source, target).execute()
    assert not target.exists()


def test_pilot_conditions_and_dynamic_dataset(tmp_path):
    conditions = pilot_conditions()
    assert len({c["id"] for c in conditions}) == 8
    assert [c for c in conditions if c["id"] == "lora-r100"][0]["rank"] == 100
    config = {"checkpoint": "f.pt", "checkpoint_sha256": "hash", "dataset_selection_sha256": "split",
              "dataset_name": "DIOR"}
    RunQueuedTransferStudy._declare_study(config, tmp_path, conditions, [1])
    assert json.loads((tmp_path / "study.json").read_text())["dataset"] == "DIOR"


def test_label_capacity_validation(tmp_path):
    request = AdapterExperimentRequest("yolox-l", tmp_path, tmp_path, tmp_path, ("lora",), (1,))
    assert request.max_labels == 50
    replace(request, max_labels=596, max_detections=1000).validate()
    with pytest.raises(MLXUserError):
        replace(request, max_labels=0).validate()


def test_pilot_aggregate_retains_dataset_identity(tmp_path):
    output = tmp_path / "rsod"
    (output / "aggregate").mkdir(parents=True)
    (output / "aggregate/transfer-results.json").write_text('[{"method":"lora","mAP50":0.4}]')
    RunRemoteAdapterPilot._aggregate({"output": str(tmp_path), "interpretation": "pilot",
        "studies": [{"dataset": "rsod", "config": str(output / "plan.json")}]})
    rows = json.loads((tmp_path / "aggregate/results.json").read_text())["runs"]
    assert rows[0]["dataset_id"] == "rsod"


def test_pilot_stops_on_child_failure_and_requires_explicit_resume(tmp_path):
    class FailedStudy:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self):
            raise MLXUserError("CUDA unavailable")

    config = tmp_path / "pilot.json"
    config.write_text(json.dumps({"output": str(tmp_path), "total_runs": 8,
        "studies": [{"dataset": "rsod", "config": "plan.json", "runs": 8}]}))
    command = RunRemoteAdapterPilot(config, study_runner=FailedStudy)
    with pytest.raises(MLXUserError, match="CUDA unavailable"):
        command.execute()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    with pytest.raises(MLXUserError, match="explicit --resume"):
        command.execute()
