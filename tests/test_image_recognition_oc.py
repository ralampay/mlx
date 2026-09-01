from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from mlx.cli import build_parser
from mlx.cli_routing import resolve_mode_descriptor
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.algorithms import (
    DeepSVDDAlgorithm,
    OneClassAlgorithmRegistry,
)
from mlx.modes.image_recognition_oc.artifacts import (
    load_image_one_class_checkpoint,
    load_raw_checkpoint,
)
from mlx.modes.image_recognition_oc.data import OneClassImageDataset
from mlx.modes.image_recognition_oc.evaluation import BenchmarkImageOneClass
from mlx.modes.image_recognition_oc.inference import InferImageOneClass
from mlx.modes.image_recognition_oc.list_models import ListImageOneClassModels
from mlx.modes.image_recognition_oc.requests import (
    BenchmarkImageOneClassRequest,
    InferImageOneClassRequest,
    ListImageOneClassModelsRequest,
    TrainImageOneClassRequest,
)
from mlx.modes.image_recognition_oc.training import TrainImageOneClassModel


class TinyBackbone(nn.Module):
    feature_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, images):
        return self.projection(images.reshape(images.shape[0], 4))


def tiny_registry() -> OneClassAlgorithmRegistry:
    return OneClassAlgorithmRegistry().register(
        DeepSVDDAlgorithm(backbone_factory=lambda _name, _config: TinyBackbone())
    )


class SyntheticImages(Dataset):
    def __init__(self, _path, *, split, normal_only, **_kwargs) -> None:
        values = [0.1, 0.2, 0.3, 0.4] if normal_only else [0.1, 0.2, 2.0, 3.0]
        labels = [0] * len(values) if normal_only else [0, 0, 1, 1]
        self.samples = [
            (torch.full((1, 2, 2), value), torch.tensor(label), f"{split}/{index}.png")
            for index, (value, label) in enumerate(zip(values, labels, strict=True))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_mode_descriptor_and_backbone_cli_option() -> None:
    descriptor = resolve_mode_descriptor("image-recognition-oc")
    assert descriptor.name == "image_recognition_oc"
    assert descriptor.default_action == "ls-models"
    assert descriptor.actions == (
        "train",
        "train-all",
        "infer-image",
        "benchmark",
        "resume",
        "status",
        "stop",
        "ls-models",
    )
    assert build_parser().parse_args(["--backbone", "resnet18"]).backbone == "resnet18"
    assert build_parser().parse_args(["--no-colored"]).colored is False


def test_deep_svdd_model_scores_and_threshold_boundary() -> None:
    algorithm = tiny_registry().get("deep-svdd")
    model = algorithm.build_model("tiny", {"svdd_dim": 2, "svdd_hidden_dim": 3})
    output = model(torch.ones(2, 1, 2, 2))
    assert output.embedding.shape == (2, 2)
    assert output.anomaly_score.shape == (2,)
    model.threshold.copy_(output.anomaly_score[0])
    assert model.classify(torch.ones(1, 1, 2, 2)).item() is False


def test_algorithm_registry_is_immutable_and_reports_unknown_model() -> None:
    registry = tiny_registry()
    with pytest.raises(TypeError):
        registry.algorithms["other"] = registry.get("deep-svdd")
    with pytest.raises(MLXUserError, match="Unsupported one-class model"):
        registry.get("other")


def _write_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(value, value, value)).save(path)


def test_dataset_contract_and_anomaly_leakage(tmp_path) -> None:
    _write_image(tmp_path / "train" / "normal" / "nested" / "one.png")
    dataset = OneClassImageDataset(
        tmp_path,
        split="train",
        height=4,
        width=4,
        colored=True,
        normal_only=True,
    )
    tensor, label, path = dataset[0]
    assert tensor.shape == (3, 4, 4)
    assert label.item() == 0
    assert path.endswith("one.png")

    _write_image(tmp_path / "train" / "anomaly" / "bad.png")
    with pytest.raises(MLXUserError, match="Anomalous images"):
        OneClassImageDataset(
            tmp_path,
            split="train",
            height=4,
            width=4,
            colored=True,
            normal_only=True,
        )


def test_benchmark_dataset_requires_both_labels(tmp_path) -> None:
    _write_image(tmp_path / "test" / "normal" / "one.png")
    with pytest.raises(MLXUserError, match="both normal and anomaly"):
        OneClassImageDataset(
            tmp_path,
            split="test",
            height=4,
            width=4,
            colored=True,
            normal_only=False,
        )


def training_request(tmp_path, **changes) -> TrainImageOneClassRequest:
    values = {
        "model": "deep-svdd",
        "backbone": "tiny",
        "dataset_path": "unused",
        "output_path": str(tmp_path / "artifacts"),
        "device": "cpu",
        "width": 2,
        "height": 2,
        "batch_size": 2,
        "workers": 0,
        "epochs": 1,
        "lr": 0.001,
        "svdd_dim": 2,
        "svdd_hidden_dim": 3,
        "svdd_quantile": 0.75,
        "random_seed": 7,
    }
    values.update(changes)
    return TrainImageOneClassRequest(**values)


def test_training_calibrates_deployment_and_resumable_checkpoints(tmp_path) -> None:
    command = TrainImageOneClassModel(
        training_request(tmp_path),
        registry=tiny_registry(),
        dataset_factory=SyntheticImages,
    )
    result = command.execute()
    deployment = load_raw_checkpoint(result["paths"]["checkpoint"])
    resumable = load_raw_checkpoint(result["paths"]["last_checkpoint"])
    assert result["paths"]["checkpoint"].name == "tiny-deep-svdd.pth"
    assert deployment["threshold"] is not None
    assert resumable["threshold"] is not None
    assert resumable["completed_epoch"] == 1
    assert "optimizer_state_dict" in resumable
    assert len(result["history"]) == 1

    resumed = TrainImageOneClassModel(
        training_request(
            tmp_path,
            epochs=2,
            model_path=str(result["paths"]["last_checkpoint"]),
        ),
        registry=tiny_registry(),
        dataset_factory=SyntheticImages,
    ).execute()
    assert load_raw_checkpoint(resumed["paths"]["last_checkpoint"])["completed_epoch"] == 2


def test_checkpoint_loader_rejects_non_finite_center(tmp_path) -> None:
    algorithm = tiny_registry().get("deep-svdd")
    model = algorithm.build_model("tiny", {"svdd_dim": 2, "svdd_hidden_dim": 3})
    model.center.fill_(float("nan"))
    model.threshold.fill_(0.5)
    checkpoint = {
        "checkpoint_version": 1,
        "mode": "image_recognition_oc",
        "model_name": "deep-svdd",
        "backbone_name": "tiny",
        "backbone_class": "TinyBackbone",
        "backbone_feature_dim": 4,
        "input_height": 2,
        "input_width": 2,
        "colored": True,
        "drax_fusion_mode": "average",
        "svdd_dim": 2,
        "svdd_hidden_dim": 3,
        "svdd_quantile": 0.95,
        "threshold": 0.5,
        "score_type": "squared_euclidean",
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "bad.pth"
    torch.save(checkpoint, path)
    with pytest.raises(MLXUserError, match="non-finite"):
        load_image_one_class_checkpoint(
            path,
            device="cpu",
            registry=tiny_registry(),
        )


class ScoreAlgorithm:
    def scores(self, _model, images):
        return images.reshape(images.shape[0], -1).mean(dim=1)


class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))


def fake_checkpoint_loader(*_args, **_kwargs):
    model = EmptyModel()
    checkpoint = {
        "model_name": "deep-svdd",
        "backbone_name": "tiny",
        "threshold": 0.5,
        "score_type": "squared_euclidean",
        "svdd_quantile": 0.95,
        "backbone_class": "TinyBackbone",
    }
    stored = {"height": 2, "width": 2, "colored": True}
    return model, checkpoint, stored, ScoreAlgorithm()


def test_single_image_inference_returns_structured_verdict() -> None:
    command = InferImageOneClass(
        InferImageOneClassRequest(
            model=None,
            backbone=None,
            model_path="model.pth",
            input_img="sample.png",
        ),
        checkpoint_loader=fake_checkpoint_loader,
        image_loader=lambda *_args, **_kwargs: torch.ones(1, 2, 2),
    )
    result = command.execute()
    assert result.is_anomaly is True
    assert result.predicted_label == "anomaly"
    assert result.anomaly_score == pytest.approx(1.0)


def test_benchmark_writes_research_artifacts(tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "benchmark"
    result = BenchmarkImageOneClass(
        BenchmarkImageOneClassRequest(
            model=None,
            backbone=None,
            model_path=str(checkpoint),
            dataset_path=str(tmp_path / "dataset"),
            output_path=str(output),
            batch_size=2,
            workers=0,
            plots=False,
        ),
        checkpoint_loader=fake_checkpoint_loader,
        dataset_factory=SyntheticImages,
    ).execute()
    assert result["metrics"]["auroc"] == pytest.approx(1.0)
    assert (output / "metrics.json").is_file()
    assert (output / "predictions.jsonl").is_file()
    assert (output / "benchmark_report.md").is_file()
    assert not (output / "roc_curve.png").exists()


def test_model_listing_uses_only_injected_backbone_inventory() -> None:
    summaries = ListImageOneClassModels(
        ListImageOneClassModelsRequest(
            model="deep-svdd",
            svdd_dim=2,
            svdd_hidden_dim=3,
        ),
        registry=tiny_registry(),
        backbone_names=lambda: ("tiny",),
    ).execute()
    assert [(item.model_name, item.backbone_name) for item in summaries] == [
        ("deep-svdd", "tiny")
    ]
