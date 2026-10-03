import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
from mlx.modes.object_detection.adapter_metrics import measure_precision_recall
from mlx.modes.object_detection.adapter_data import prepare_adapter_dataset
from mlx.cli import build_parser


def test_adapter_config_defaults_and_validation(tmp_path):
    request = AdapterExperimentRequest.from_config({"model": "yolox-l", "dataset_path": str(tmp_path),
                                                     "output_path": str(tmp_path / "out")})
    assert request.methods == ("drax",)
    assert request.checkpoint.name == "foundational-yolox-l.pt"
    with pytest.raises(MLXUserError):
        AdapterExperimentRequest.from_config({"model": "yolox-l", "methods": "made-up"})
    options = build_parser().parse_args(["--mode", "object-detection", "--action", "adapter-experiment",
        "--checkpoint", "foundation.pt", "--methods", "frozen,drax", "--adapter-target", "neck",
        "--adapter-rank", "4", "--experiment-seeds", "1,2,3"])
    assert options.model_path == "foundation.pt"
    assert options.methods == "frozen,drax"
    assert options.experiment_seeds == "1,2,3"


def test_report_serialization_and_paired_summary(tmp_path):
    for method, value in (("drax", .4), ("bottleneck", .3)):
        folder = tmp_path / method / "seed-42"
        folder.mkdir(parents=True)
        (folder / "metrics.json").write_text(json.dumps({
            "method": method, "seed": 42, "map_50": value, "map_50_95": value,
            "precision": value, "recall": value, "trainable_params": 10,
            "total_params": 100, "trainable_percent": 10, "training_seconds": 1,
            "peak_memory_bytes": None, "checkpoint_size_bytes": 20,
            "inference_latency_ms": 5}))
    result = GenerateAdapterReport(tmp_path).execute()
    assert result["runs"] == 2
    assert len(list(csv.DictReader(Path(result["csv"]).open()))) == 2
    assert "Drax minus bottleneck" in Path(result["report"]).read_text()


def test_fixed_threshold_precision_recall(tmp_path):
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(image_dir / "sample.png")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    class FakeBoxes:
        xyxy = torch.tensor([[40., 40., 60., 60.], [0., 0., 10., 10.]])
        cls = torch.tensor([0, 0])
        conf = torch.tensor([.9, .8])

    class FakeModel:
        def predict(self, **kwargs):
            return type("Result", (), {"boxes": FakeBoxes()})()

    values = measure_precision_recall(FakeModel(), tmp_path, image_size=640)
    assert values["precision"] == .5
    assert values["recall"] == 1.0


def test_existing_dataset_splits_are_preserved(tmp_path):
    source = tmp_path / "source"
    (source / "data.yaml").parent.mkdir(parents=True)
    (source / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [car]\n")
    for split in ("train", "val", "test"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (10, 10)).save(image_dir / f"{split}-{index}.png")
            (label_dir / f"{split}-{index}.txt").write_text("0 .5 .5 .5 .5\n")
    result = prepare_adapter_dataset(source, tmp_path / "output")
    assert {split: row["images"] for split, row in result["splits"].items()} == {
        "train": 10, "val": 10, "test": 10}
    assert (tmp_path / "output/images/train/train-0.png").is_symlink()
