import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_data import (
    FOUNDATION_CLASSES,
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


def test_adapter_dataset_explicit_path():
    explicit = AdapterExperimentRequest.from_config({"dataset_path": "/custom", "_explicit_options": {"dataset_path"}})
    assert explicit.dataset == Path("/custom")


@pytest.mark.parametrize("adapter_method", ["drax-hybrid", "drax-residual-fusion"])
@pytest.mark.parametrize("matching_placement", [False, True])
def test_resume_checks_hybrid_injection_placement(tmp_path, monkeypatch, matching_placement, adapter_method):
    from mlx.modes.object_detection import adapter_experiment as experiment
    from mlx.modes.object_detection.libreyolo import adapter_targets

    request = AdapterExperimentRequest.from_config({
        "output_path": str(tmp_path / "output"), "dataset_path": str(tmp_path / "data"),
        "model_path": str(tmp_path / "foundation.pt"), "adapter": adapter_method,
        "device": "cpu", "epochs": 20,
    })
    foundation = {"classes": {0: "car"}, "nc": 1, "sha256": "foundation"}
    dataset = {"classes": ["car"], "dataset": "fixture", "selection_sha256": "split"}
    monkeypatch.setattr(experiment.VerifyFoundationCheckpoint, "execute", lambda self: (torch.nn.Identity(), foundation))
    monkeypatch.setattr(experiment.CollectAdapterEnvironment, "execute", lambda self: {})
    monkeypatch.setattr(experiment, "load_prepared_adapter_dataset", lambda path: dataset)
    monkeypatch.setattr(adapter_targets, "yolox_targets", lambda *args, **kwargs: {"neck.conv": 4})

    def unexpected_training(*args):
        raise AssertionError("Resume validation must not start training")

    monkeypatch.setattr(experiment.RunAdapterExperiment, "_run_one", unexpected_training)
    for method in ("frozen", adapter_method):
        hybrid = method == adapter_method
        metrics = {
            "status": "completed", "checkpoint_sha256": "foundation",
            "dataset_selection_sha256": "split", "physical_batch_size": 1,
            "effective_batch_size": 1, "image_size": 640, "device": "cpu", "amp": True,
            "epochs": 20 if hybrid else 0, "method": method, "seed": 42,
            "learning_rate": .0001, "adapter_rank": 8 if hybrid and adapter_method == "drax-hybrid" else None,
            "adapter_reduction": 8 if hybrid else None, "adapter_alpha": 1.0 if hybrid else None,
            "train_head": False, "adapter_target": "neck" if hybrid else None,
            "injected_modules": ["neck.conv" if matching_placement else "old.conv"] if hybrid else [],
        }
        directory = request.output / method / "seed-42"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps(metrics))
    command = experiment.RunAdapterExperiment(request, dataset_loader=lambda path: dataset)
    if matching_placement:
        assert len(command.execute()) == 2
    else:
        with pytest.raises(MLXUserError, match="incompatible"):
            command.execute()


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


def test_freezing_verification_detects_batchnorm_buffer_updates(tmp_path):
    checkpoint = tmp_path / "last.pt"
    torch.save({"train_model": {"bn.running_mean": torch.ones(2), "adapter": torch.ones(1)}}, checkpoint)
    changed, learned = _verify_training_checkpoint(
        {"last_checkpoint": str(checkpoint)},
        {"bn.running_mean": torch.zeros(2)}, {"adapter": torch.zeros(1)},
    )
    assert changed == [("bn.running_mean", 1.0)]
    assert learned


def test_residual_fusion_cli_configuration():
    options = build_parser().parse_args([
        '--mode', 'object-detection', '--action', 'adapter-experiment',
        '--adapter', 'drax-residual-fusion', '--adapter-reduction', '16',
        '--adapter-alpha', '1', '--adapter-target', 'neck',
    ])
    request = AdapterExperimentRequest.from_config(vars(options))
    assert request.methods == ('drax-residual-fusion',)
    assert request.reduction == 16
    assert request.alpha == 1
    assert request.target == 'neck'
