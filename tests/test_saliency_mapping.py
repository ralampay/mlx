from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mlx.cli_config import build_runtime_config
from mlx.cli_routing import resolve_mode_descriptor
from mlx.modes.saliency_mapping import benchmark as saliency_benchmark
from mlx.modes.saliency_mapping import train as saliency_train
from mlx.modes.saliency_mapping.benchmark import BenchmarkSaliencyMapping, BenchmarkSaliencyModelGroup
from mlx.modes.saliency_mapping.checkpoints import checkpoint_payload
from mlx.modes.saliency_mapping.data import SaliencyDataset
from mlx.modes.saliency_mapping.losses import SaliencyHybridLoss, iou_loss, ssim_loss
from mlx.modes.saliency_mapping.metrics import (
    SaliencyMetricAccumulator,
    f_beta,
    mean_absolute_error,
    per_image_saliency_metrics,
    saliency_threshold_sweep,
)
from mlx.modes.saliency_mapping.models import (
    MODEL_NAMES,
    SMALL_MODEL_NAMES,
    build_saliency_model,
    grouped_model_names,
    supported_model_names,
)
from mlx.modes.saliency_mapping.requests import BenchmarkSaliencyRequest, SaliencyRequest, TrainSaliencyRequest
from mlx.modes.saliency_mapping.samples import GenerateSaliencySamples
from mlx.modes.saliency_mapping.train import SmokeTestSaliencyModels, TrainSaliencyModel
from mlx.modes.segmentation.models import SMALL_MODEL_NAMES as SEGMENTATION_SMALL_MODEL_NAMES


def _dataset(root: Path, *, splits=("train", "val", "test"), count=2) -> Path:
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir()
        for index in range(count):
            ramp = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
            image = np.repeat(ramp[..., None], 3, axis=2)
            cv2.imwrite(str(root / split / "images" / f"sample_{index}.png"), image)
            cv2.imwrite(str(root / split / "masks" / f"sample_{index}.png"), ramp)
    return root


def test_both_mode_aliases_resolve_to_saliency_runner() -> None:
    canonical = resolve_mode_descriptor("saliency_mapping")
    alias = resolve_mode_descriptor("saliency-mapping")
    assert canonical is alias
    assert canonical.runner.endswith("saliency_mapping.runner:run_saliency_mapping")


def test_cli_config_normalizes_hyphenated_saliency_mode() -> None:
    namespace = Namespace(mode="saliency-mapping", width=10, height=12, help=False)
    config = build_runtime_config(namespace)
    assert config["mode"] == "saliency_mapping"
    assert config["input_size"] == (10, 12)


def test_model_groups_exactly_match_segmentation_semantics() -> None:
    assert SMALL_MODEL_NAMES == SEGMENTATION_SMALL_MODEL_NAMES
    assert grouped_model_names("all-small") == sorted(SEGMENTATION_SMALL_MODEL_NAMES)
    assert grouped_model_names("all") == supported_model_names()
    assert set(grouped_model_names("all")) == MODEL_NAMES


@pytest.mark.parametrize(
    "model_name",
    [
        "unet",
        "unet-resnet18",
        "unet-resnet50",
        "unet-densenet121",
        "unet-mobilenet_v3_large",
        "unet-efficientnet_b0",
        "unet-convnext_tiny",
        "unet-convnext_small",
        "unet-convnext_base",
        "unet-convnext_large",
        "unet-draxnet-average",
        "unet-draxnet-sknet",
        "unet-drax_mobilenet_v3_large-average",
        "unet-drax_mobilenet_v3_large-sknet",
    ],
)
def test_representative_models_return_one_channel_logits(model_name: str) -> None:
    model = build_saliency_model(model_name, {"colored": True, "pretrained": False}).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
    probability = torch.sigmoid(output)
    assert probability.min() >= 0
    assert probability.max() <= 1


def test_smoke_command_supports_specific_model() -> None:
    result = SmokeTestSaliencyModels(
        SaliencyRequest(model="unet", input_size=(32, 32), batch_size=1)
    ).execute()
    assert result[0]["output_shape"] == [1, 1, 32, 32]


def test_continuous_saliency_dataset_contract(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset", splits=("train",), count=1)
    dataset = SaliencyDataset(root, split="train", input_size=(8, 8))
    image, target = dataset[0]
    assert image.shape == (3, 8, 8)
    assert target.shape == (1, 8, 8)
    assert target.dtype == torch.float32
    assert 0 <= target.min() <= target.max() <= 1
    assert torch.unique(target).numel() > 2
    assert target[0, 0, 1] == pytest.approx(4 / 255)


def test_hybrid_loss_components_and_backpropagation() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    targets = torch.rand(2, 1, 8, 8)
    probabilities = torch.sigmoid(logits)
    assert torch.isfinite(ssim_loss(probabilities, targets))
    assert torch.isfinite(iou_loss(probabilities, targets))
    result = SaliencyHybridLoss()(logits, targets)
    assert set(result.as_dict()) == {"loss", "bce_loss", "ssim_loss", "iou_loss"}
    assert all(torch.isfinite(value) for value in result.as_dict().values())
    result.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mae_f_beta_threshold_and_precision_recall_correctness() -> None:
    target = np.asarray([[[[0.0, 0.0], [1.0, 1.0]]]])
    perfect = target.copy()
    assert mean_absolute_error(perfect, target) == 0
    assert f_beta(1, 1) == 1
    rows = saliency_threshold_sweep(perfect, target, threshold_steps=3)
    metrics, returned_rows = per_image_saliency_metrics(perfect, target, threshold_steps=3)
    assert rows == returned_rows
    assert metrics["mae"] == 0
    assert metrics["max_f_beta"] == 1
    assert metrics["best_threshold"] == pytest.approx(0.5)
    assert all({"threshold", "precision", "recall", "f_beta"} <= row.keys() for row in rows)
    accumulator = SaliencyMetricAccumulator(3)
    accumulator.update(perfect, target)
    aggregate, aggregate_rows = accumulator.finalize()
    assert aggregate["mae"] == 0
    assert aggregate["max_f_beta"] == 1
    assert aggregate_rows == rows


class _FixedSaliencyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images[:, :1] * 8 - 4) * self.scale


def test_benchmark_writes_complete_artifacts_and_samples(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path / "dataset", splits=("test",), count=2)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        saliency_benchmark,
        "load_checkpoint_bundle",
        lambda config: (
            _FixedSaliencyModel(),
            {
                "checkpoint_path": checkpoint,
                "model_name": "unet",
                "input_size": (8, 8),
                "transform": "resize",
                "colored": True,
                "output_channels": 1,
            },
        ),
    )
    output = tmp_path / "benchmark"
    metrics = BenchmarkSaliencyMapping(
        BenchmarkSaliencyRequest(
            model="unet",
            model_path=str(checkpoint),
            dataset_path=str(dataset),
            output_path=str(output),
            input_size=(8, 8),
            batch_size=1,
            threshold_steps=3,
            workers=0,
        )
    ).execute()
    assert metrics["evaluated_images"] == 2
    expected = {
        "metrics.json",
        "metrics.csv",
        "image_metrics.csv",
        "threshold_metrics.csv",
        "precision_recall.csv",
        "timing.csv",
        "run_metadata.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert len(list((output / "samples" / "panels").glob("*.png"))) == 2
    panel = cv2.imread(str(next((output / "samples" / "panels").glob("*.png"))))
    assert panel is not None and panel.shape[1] == 32


def test_sample_command_writes_expected_views(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path / "dataset", splits=("test",), count=1)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    from mlx.modes.saliency_mapping import samples
    monkeypatch.setattr(
        samples,
        "load_checkpoint_bundle",
        lambda config: (
            _FixedSaliencyModel(),
            {"model_name": "unet", "input_size": (8, 8), "transform": "resize", "colored": True},
        ),
    )
    result = GenerateSaliencySamples(
        {"device": "cpu"},
        checkpoint_path=checkpoint,
        split_path=dataset / "test",
        output_dir=tmp_path / "output",
    ).execute()
    assert len(result) == 1
    for name in ("original", "ground_truth", "prediction", "heatmap", "overlay", "panels"):
        assert len(list((tmp_path / "output" / "samples" / name).glob("*.png"))) == 1


@pytest.mark.parametrize("selector", ["all-small", "all"])
def test_group_benchmark_comparison(selector: str, tmp_path: Path) -> None:
    calls = []

    class FakeBenchmark:
        def __init__(self, request, reporter):
            self.request = request

        def execute(self):
            calls.append(self.request.model)
            return {"mae": 0.1, "max_f_beta": 0.9, "mean_f_beta": 0.8}

    output = tmp_path / selector
    result = BenchmarkSaliencyModelGroup(
        BenchmarkSaliencyRequest(
            model=selector,
            dataset_path=str(tmp_path / "dataset"),
            output_path=str(output),
        ),
        benchmark_factory=lambda request, reporter: FakeBenchmark(request, reporter),
    ).execute()
    assert calls == grouped_model_names(selector)
    assert len(result["models"]) == len(calls)
    assert (output / "comparison.csv").is_file()
    assert (output / "comparison.json").is_file()


def test_checkpoint_payload_preserves_saliency_metadata() -> None:
    payload = checkpoint_payload(
        nn.Conv2d(3, 1, 1),
        model_name="unet",
        config={"input_size": (32, 40), "transform": "center-crop", "colored": True},
    )
    assert payload["family"] == "saliency_mapping"
    assert payload["output_channels"] == 1
    assert payload["activation"] == "sigmoid"
    assert payload["model_name"] == "unet"


def test_minimal_training_smoke_writes_best_last_and_history(tmp_path: Path, monkeypatch) -> None:
    tensors = TensorDataset(torch.rand(2, 3, 8, 8), torch.rand(2, 1, 8, 8))
    monkeypatch.setattr(saliency_train, "load_saliency_datasets", lambda *args, **kwargs: (tensors, tensors))
    monkeypatch.setattr(saliency_train, "build_saliency_model", lambda *args, **kwargs: _FixedSaliencyModel())
    output = tmp_path / "training"
    result = TrainSaliencyModel(
        TrainSaliencyRequest(
            model="unet",
            dataset_path=str(tmp_path / "dataset"),
            output_path=str(output),
            input_size=(8, 8),
            batch_size=1,
            epochs=1,
            workers=0,
            threshold_steps=3,
        )
    ).execute()
    assert Path(result["checkpoint_path"]).is_file()
    assert Path(result["last_checkpoint_path"]).is_file()
    assert (output / "training.csv").is_file()
    assert (output / "training_curves.png").is_file()
    state = torch.load(result["last_checkpoint_path"], weights_only=True)
    assert state["completed_epoch"] == 1
    assert state["family"] == "saliency_mapping"


def test_model_listing_is_json_compatible() -> None:
    from mlx.core.artifacts import json_safe
    from mlx.modes.saliency_mapping.list_models import ListSaliencyModels
    value = json_safe(ListSaliencyModels({"pretrained": False}).execute())
    assert len(value) == len(MODEL_NAMES)
    assert all({"model_name", "parameter_count", "backbone", "pretrained_supported", "groups"} <= row.keys() for row in value)
