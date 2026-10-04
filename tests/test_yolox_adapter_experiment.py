import csv
from collections import Counter
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_data import (
    DAWN_CLASS_MAPPING,
    FOUNDATION_CLASSES,
    _DawnRecord,
    _stratified_partitions,
    load_prepared_adapter_dataset,
)
from mlx.modes.object_detection.adapter_experiment import (
    AdapterExperimentRequest,
    _restore_best_checkpoint,
    _verify_training_checkpoint,
)
from mlx.modes.object_detection.adapter_metrics import measure_precision_recall
from mlx.modes.object_detection.adapter_report import FIELDS, GenerateAdapterReport
from mlx.modes.object_detection.libreyolo.adapter_backend import require_experiment_device


def test_adapter_config_cli_and_validation(tmp_path):
    request = AdapterExperimentRequest.from_config(
        {"model": "yolox-l", "dataset_path": str(tmp_path), "output_path": str(tmp_path / "out")}
    )
    assert request.methods == ("drax",)
    assert request.checkpoint.name == "foundational-yolox-l.pt"
    assert request.gradient_accumulation == 1
    with pytest.raises(MLXUserError):
        AdapterExperimentRequest.from_config({"model": "yolox-l", "methods": "made-up"})
    options = build_parser().parse_args(
        [
            "--mode", "object-detection", "--action", "adapter-experiment",
            "--checkpoint", "foundation.pt", "--methods", "frozen,drax",
            "--adapter-target", "neck", "--adapter-rank", "4",
            "--experiment-seeds", "1,2,3", "--gradient-accumulation", "4",
            "--device", "cuda", "--no-amp",
        ]
    )
    assert options.model_path == "foundation.pt"
    assert options.methods == "frozen,drax"
    assert options.experiment_seeds == "1,2,3"
    assert options.gradient_accumulation == 4
    assert options.device == "cuda"
    assert options.amp is False


def test_cuda_required_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MLXUserError, match="CUDA was requested but is not available.*not started"):
        require_experiment_device("cuda")
    assert require_experiment_device("cpu") == torch.device("cpu")


def _record(index: int) -> _DawnRecord:
    class_id = index % len(FOUNDATION_CLASSES)
    source_name = next(name for name, mapped in DAWN_CLASS_MAPPING.items() if mapped == class_id)
    return _DawnRecord(
        image_id=f"foggy-{index:04d}", image_bytes=b"image", image_suffix=".jpg",
        width=10, height=10,
        objects=({"class_name": source_name},), source_split="train",
    )


def test_dataset_mapping_and_subset_are_reproducible():
    assert DAWN_CLASS_MAPPING == {
        "Person": 0, "Bicycle": 1, "Motorcycle": 2,
        "Car": 3, "Bus": 4, "Truck": 5,
    }
    records = [_record(index) for index in range(120)]
    first = _stratified_partitions(records, 42)
    second = _stratified_partitions(records, 42)
    assert {key: [row.image_id for row in value] for key, value in first.items()} == {
        key: [row.image_id for row in value] for key, value in second.items()
    }
    assert {key: len(value) for key, value in first.items()} == {"train": 84, "val": 18, "test": 18}
    for rows in first.values():
        counts = sum((row.class_counts for row in rows), Counter())
        assert all(counts[class_id] for class_id in range(6))


def test_prepared_dataset_validation(tmp_path):
    manifest = {"classes": list(FOUNDATION_CLASSES)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "data.yaml").write_text("names: []\n")
    for split in ("train", "val", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    assert load_prepared_adapter_dataset(tmp_path)["classes"] == list(FOUNDATION_CLASSES)


def test_report_serialization_output_directories_and_paired_summary(tmp_path):
    (tmp_path / "environment.json").write_text(json.dumps({"gpu_name": "test-gpu"}))
    for method, value in (("drax", 0.4), ("bottleneck", 0.3)):
        folder = tmp_path / method / "seed-42"
        folder.mkdir(parents=True)
        values = {field: None for field in FIELDS}
        values.update(
            {
                "experiment_id": f"{method}-seed-42", "method": method, "seed": 42,
                "mAP50": value, "mAP50_95": value, "precision": value, "recall": value,
                "trainable_params": 10, "total_params": 100, "trainable_percent": 10,
                "training_seconds": 1, "peak_cuda_memory_mb": 20,
                "checkpoint_size_mb": 1, "inference_latency_ms": 5, "status": "completed",
            }
        )
        (folder / "metrics.json").write_text(json.dumps(values))
    result = GenerateAdapterReport(tmp_path).execute()
    assert result["runs"] == 2
    assert Path(result["csv"]).name == "results.csv"
    assert Path(result["json"]).name == "results.json"
    assert Path(result["summary"]).name == "summary.md"
    rows = list(csv.DictReader(Path(result["csv"]).open()))
    assert len(rows) == 2 and all(row["gpu"] == "test-gpu" for row in rows)
    assert "drax minus bottleneck" in Path(result["summary"]).read_text()


def test_fixed_threshold_precision_recall(tmp_path):
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(image_dir / "sample.png")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    class FakeBoxes:
        xyxy = torch.tensor([[40.0, 40.0, 60.0, 60.0], [0.0, 0.0, 10.0, 10.0]])
        cls = torch.tensor([0, 0])
        conf = torch.tensor([0.9, 0.8])

    class FakeModel:
        def predict(self, **kwargs):
            return type("Result", (), {"boxes": FakeBoxes()})()

    values = measure_precision_recall(FakeModel(), tmp_path, image_size=640)
    assert values["precision"] == 0.5
    assert values["recall"] == 1.0


def test_best_validation_checkpoint_is_restored_before_test_evaluation(tmp_path):
    model = torch.nn.Linear(2, 1)
    expected = {name: torch.full_like(value, 3.0) for name, value in model.state_dict().items()}
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": expected,
            "best_epoch": 7,
            "best_mAP50": 0.6,
            "best_mAP50_95": 0.4,
        },
        checkpoint,
    )
    path, selection = _restore_best_checkpoint(model, {"best_checkpoint": str(checkpoint)})
    assert path == checkpoint
    assert selection["selected_epoch"] == 7
    assert selection["selected_validation_mAP50_95"] == 0.4
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_freezing_is_verified_against_non_ema_training_state(tmp_path):
    checkpoint = tmp_path / "last.pt"
    frozen = torch.tensor([1.0])
    trainable = torch.tensor([2.0])
    torch.save(
        {
            "model": {"frozen": frozen + 1e-6, "trainable": trainable + 1.0},
            "train_model": {"frozen": frozen.clone(), "trainable": trainable + 1.0},
        },
        checkpoint,
    )
    changed_frozen, trainable_changed = _verify_training_checkpoint(
        {"last_checkpoint": str(checkpoint)},
        {"frozen": frozen},
        {"trainable": trainable},
    )
    assert changed_frozen == []
    assert trainable_changed is True
