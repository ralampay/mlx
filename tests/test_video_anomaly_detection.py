from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from mlx.cli import build_parser
from mlx.cli_routing import MODE_REGISTRY
from mlx.core.artifacts import json_safe
from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import FrameSourceMetadata
from mlx.modes.image_classification.models import (
    ONE_SHOT_MODEL_NAMES,
    STANDARD_MODEL_NAMES,
    build_image_feature_backbone,
    standard_model_names,
)
from mlx.modes.video_anomaly_detection.clips import (
    aggregate_frame_scores,
    window_start_indices,
)
from mlx.modes.video_anomaly_detection.data import VideoClipDataset
from mlx.modes.video_anomaly_detection.metrics import compute_binary_metrics
from mlx.modes.video_anomaly_detection.models import (
    BACKBONE_3D_REGISTRY,
    DeepSVDDHead,
    FrameBackbone,
    TemporalConvEncoder,
    VideoAnomaly3DModel,
    VideoAnomalyModel,
    build_spatiotemporal_backbone_3d,
    build_video_anomaly_model,
)
from mlx.modes.video_anomaly_detection.models.backbone3d import inflate_conv2d
from mlx.modes.video_anomaly_detection.artifacts import (
    artifact_stem,
    config_from_checkpoint,
    load_video_anomaly_checkpoint,
    model_metadata,
    save_deployment_checkpoint,
)
from mlx.modes.video_anomaly_detection.requests import TrainVideoAnomalyRequest
from mlx.modes.video_anomaly_detection.requests import (
    BenchmarkVideoAnomalyRequest,
    InferVideoAnomalyRequest,
    ListVideoAnomalyModelsRequest,
)
from mlx.modes.video_anomaly_detection.evaluation import BenchmarkVideoAnomalyModel
from mlx.modes.video_anomaly_detection.inference import InferVideoAnomaly
from mlx.modes.video_anomaly_detection.list_models import ListVideoAnomalyModels
from mlx.modes.video_anomaly_detection.presentation import annotate_video_anomaly_frame
from mlx.modes.video_anomaly_detection.training import TrainVideoAnomalyModel


class TinyFeatureBackbone(nn.Module):
    feature_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))


def tiny_video_model(*_, **__) -> VideoAnomalyModel:
    return VideoAnomalyModel(
        FrameBackbone(TinyFeatureBackbone()),
        TemporalConvEncoder(4, 5, 3, kernel_size=3),
        DeepSVDDHead(3, 4, 2),
    )


class Tiny3DBackbone(nn.Module):
    feature_dim = 4
    temporal_kernel_size = 3
    temporal_stride_policy = "preserve"
    pooling = "adaptive_avg_3d"
    pretrained_provenance = "none"

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=1)

    def forward(self, clips):
        return self.conv(clips.transpose(1, 2)).mean((2, 3, 4))


def tiny_3d_video_model(*_, **__) -> VideoAnomaly3DModel:
    return VideoAnomaly3DModel(Tiny3DBackbone(), DeepSVDDHead(4, 5, 2))


def test_video_model_shape_flow_and_squared_distance() -> None:
    model = tiny_video_model()
    model.svdd_head.center.copy_(torch.tensor([1.0, -2.0]))
    output = model(torch.randn(2, 4, 3, 8, 8))

    assert output.frame_features.shape == (2, 4, 4)
    assert output.clip_embedding.shape == (2, 3)
    assert output.svdd_embedding.shape == (2, 2)
    assert output.anomaly_score.shape == (2,)
    assert torch.equal(
        output.anomaly_score,
        ((output.svdd_embedding - model.svdd_head.center) ** 2).sum(dim=1),
    )
    assert model.svdd_head.center.grad is None
    assert "svdd_head.center" not in dict(model.named_parameters())


def test_frame_backbone_batches_all_frames_without_changing_order() -> None:
    class IdentityFeatures(nn.Module):
        feature_dim = 3

        def forward(self, images):
            return images[:, :, 0, 0]

    clips = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 1, 1)
    assert torch.equal(FrameBackbone(IdentityFeatures())(clips), clips[..., 0, 0])


def test_clip_native_3d_model_shape_flow_and_score() -> None:
    model = tiny_3d_video_model()
    model.svdd_head.center.copy_(torch.tensor([0.5, -0.5]))
    output = model(torch.randn(2, 5, 3, 8, 8))

    assert output.frame_features is None
    assert output.clip_embedding.shape == (2, 4)
    assert output.svdd_embedding.shape == (2, 2)
    assert output.anomaly_score.shape == (2,)
    assert torch.equal(
        output.anomaly_score,
        ((output.svdd_embedding - model.svdd_head.center) ** 2).sum(dim=1),
    )


def test_conv_inflation_preserves_spatial_weights_and_temporal_stride() -> None:
    source = nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1, bias=False)
    target = inflate_conv2d(source, 3)

    assert target.kernel_size == (3, 3, 3)
    assert target.stride == (1, 2, 2)
    assert torch.allclose(target.weight.sum(dim=2), source.weight)


@pytest.mark.parametrize("model_name", sorted(STANDARD_MODEL_NAMES))
def test_every_standard_alias_has_a_clip_native_3d_backbone(model_name) -> None:
    assert model_name in BACKBONE_3D_REGISTRY
    backbone = build_spatiotemporal_backbone_3d(
        model_name,
        {"pretrained": False, "backbone_temporal_kernel_size": 3},
    )
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 3, 32, 32))
    assert features.shape == (1, backbone.feature_dim)
    assert all(
        module.stride[0] == 1
        for module in backbone.modules()
        if isinstance(module, nn.Conv3d)
    )
    assert all(
        module.stride[0] == 1
        for module in backbone.modules()
        if isinstance(module, (nn.MaxPool3d, nn.AvgPool3d))
    )
    assert not any(
        isinstance(
            module,
            (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d),
        )
        for module in backbone.modules()
    )
    del backbone
    gc.collect()


@pytest.mark.parametrize("model_name", ["draxnet", "drax_mobilenet_v3_large"])
@pytest.mark.parametrize("fusion_mode", ["average", "sknet"])
def test_drax_3d_fusion_variants(model_name, fusion_mode) -> None:
    backbone = build_spatiotemporal_backbone_3d(
        model_name,
        {
            "pretrained": False,
            "backbone_temporal_kernel_size": 3,
            "drax_fusion_mode": fusion_mode,
        },
    )
    assert any(
        getattr(module, "fusion_mode", None) == fusion_mode
        for module in backbone.modules()
    )


def test_model_listing_includes_both_drax_fusions_and_no_siamese() -> None:
    summaries = ListVideoAnomalyModels(
        ListVideoAnomalyModelsRequest(),
        backbone_factory=lambda *_args: Tiny3DBackbone(),
    ).execute()
    assert not any(item.model_name.startswith("siamese-") for item in summaries)
    for name in ("draxnet", "drax_mobilenet_v3_large"):
        assert {
            item.drax_fusion_mode for item in summaries if item.model_name == name
        } == {"average", "sknet"}


def test_standard_registry_capability_excludes_every_siamese_alias() -> None:
    assert set(standard_model_names()) == STANDARD_MODEL_NAMES
    assert set(standard_model_names()).isdisjoint(ONE_SHOT_MODEL_NAMES)
    assert not any(name.startswith("siamese-") for name in standard_model_names())
    with pytest.raises(MLXUserError, match="one-shot/Siamese"):
        build_spatiotemporal_backbone_3d(
            "siamese-resnet18",
            {"pretrained": False, "backbone_temporal_kernel_size": 3},
        )


@pytest.mark.parametrize("model_name", sorted(STANDARD_MODEL_NAMES))
def test_every_standard_alias_converts_to_a_headless_feature_backbone(model_name) -> None:
    backbone = build_image_feature_backbone(
        model_name, {"colored": True, "pretrained": False}
    )
    assert backbone.feature_dim > 0
    assert not any(name.startswith("siamese-") for name, _ in backbone.named_modules())
    del backbone
    gc.collect()


def _write_sequence(root: Path, split: str, label: str, source: str, count: int) -> None:
    directory = root / split / label / source
    directory.mkdir(parents=True)
    for index in range(1, count + 1):
        Image.fromarray(np.full((6, 7, 3), index, dtype=np.uint8)).save(
            directory / f"{index:03d}.tif"
        )


def test_clip_dataset_windows_stride_order_metadata_and_incomplete_policy(tmp_path) -> None:
    _write_sequence(tmp_path, "train", "normal", "clip-b", 6)
    _write_sequence(tmp_path, "train", "normal", "clip-a", 5)
    dataset = VideoClipDataset(
        tmp_path,
        split="train",
        clip_length=3,
        frame_stride=2,
        height=8,
        width=9,
        normal_only=True,
    )

    assert window_start_indices(4, clip_length=3, frame_stride=2) == ()
    assert len(dataset) == 3
    clip, label, metadata = dataset[0]
    assert clip.shape == (3, 3, 8, 9)
    assert label.item() == 0
    assert metadata == {
        "source": "normal/clip-a",
        "start_frame": 1,
        "end_frame": 5,
        "frame_indices": [1, 3, 5],
    }


def test_normal_only_dataset_rejects_training_anomaly_sources(tmp_path) -> None:
    _write_sequence(tmp_path, "train", "normal", "normal-clip", 3)
    _write_sequence(tmp_path, "train", "anomaly", "bad-clip", 3)
    with pytest.raises(MLXUserError, match="Anomalous samples"):
        VideoClipDataset(
            tmp_path,
            split="train",
            clip_length=2,
            frame_stride=1,
            height=8,
            width=8,
            normal_only=True,
        )


class SyntheticNormalClips(Dataset):
    def __init__(self, _path, *, split, clip_length, height, width, **_kwargs) -> None:
        generator = torch.Generator().manual_seed(1 if split == "train" else 2)
        self.clips = torch.randn(4, clip_length, 3, height, width, generator=generator)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], torch.tensor(0), {
            "source": "synthetic",
            "start_frame": index,
            "end_frame": index + self.clips.shape[1] - 1,
            "frame_indices": list(range(index, index + self.clips.shape[1])),
        }


def test_normal_only_training_center_calibration_checkpoint_csv_and_resume(tmp_path) -> None:
    output = tmp_path / "artifacts"
    base = dict(
        model="resnet18",
        dataset_path="unused",
        output_path=str(output),
        device="cpu",
        width=4,
        height=4,
        batch_size=2,
        workers=0,
        clip_length=3,
        frame_stride=1,
        temporal_model="tcn",
        temporal_hidden_dim=5,
        temporal_embedding_dim=3,
        temporal_kernel_size=3,
        svdd_dim=2,
        svdd_hidden_dim=4,
        svdd_quantile=0.75,
        lr=0.001,
        random_seed=9,
        backbone_mode="frame-2d",
    )
    events = []
    result = TrainVideoAnomalyModel(
        TrainVideoAnomalyRequest(**base, epochs=1),
        reporter=CallbackWorkflowReporter(events.append),
        model_factory=tiny_video_model,
        dataset_factory=SyntheticNormalClips,
    ).execute()
    checkpoint_path = result["paths"]["checkpoint"]
    last_path = result["paths"]["last_checkpoint"]
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    last = torch.load(last_path, weights_only=True)

    assert checkpoint["svdd_center"].shape == (2,)
    assert checkpoint["svdd_threshold"] is not None
    assert last["completed_epoch"] == 1
    assert last["training_state_version"] == 1
    with (output / "training.csv").open(newline="", encoding="utf-8") as source:
        assert len(list(csv.DictReader(source))) == 1

    progress_events = [
        event
        for event in events
        if isinstance(event.payload, dict)
        and event.payload.get("event") == "video_anomaly_training_progress"
    ]
    phases = {event.payload["phase"] for event in progress_events}
    assert {
        "center",
        "train",
        "validation",
        "best-calibration",
        "last-calibration",
    } <= phases
    assert any(
        isinstance(event.payload, dict)
        and event.payload.get("event") == "video_anomaly_training_epoch"
        for event in events
    )
    for phase in phases:
        statuses = [
            event.payload["status"]
            for event in progress_events
            if event.payload["phase"] == phase
        ]
        assert statuses[0] == "start"
        assert statuses[-1] == "complete"

    resumed = TrainVideoAnomalyModel(
        TrainVideoAnomalyRequest(**base, epochs=2, model_path=str(last_path)),
        model_factory=tiny_video_model,
        dataset_factory=SyntheticNormalClips,
    ).execute()
    resumed_last = torch.load(resumed["paths"]["last_checkpoint"], weights_only=True)
    assert resumed_last["completed_epoch"] == 2
    assert len(resumed_last["history"]) == 2
    assert torch.equal(resumed_last["svdd_center"], last["svdd_center"])


def test_clip_native_3d_training_calibration_and_resume(tmp_path) -> None:
    output = tmp_path / "three-d"
    base = dict(
        model="resnet18",
        backbone_mode="3d",
        backbone_temporal_kernel_size=3,
        dataset_path="unused",
        output_path=str(output),
        device="cpu",
        width=4,
        height=4,
        batch_size=2,
        workers=0,
        clip_length=3,
        frame_stride=1,
        svdd_dim=2,
        svdd_hidden_dim=5,
        svdd_quantile=0.75,
        lr=0.001,
        random_seed=11,
    )
    first = TrainVideoAnomalyModel(
        TrainVideoAnomalyRequest(**base, epochs=1),
        model_factory=tiny_3d_video_model,
        dataset_factory=SyntheticNormalClips,
    ).execute()
    checkpoint = torch.load(first["paths"]["checkpoint"], weights_only=True)
    assert first["paths"]["checkpoint"].name == "resnet18-3d-svdd.pth"
    assert checkpoint["backbone_mode"] == "3d"
    assert checkpoint["svdd_threshold"] is not None

    resumed = TrainVideoAnomalyModel(
        TrainVideoAnomalyRequest(
            **base,
            epochs=2,
            model_path=str(first["paths"]["last_checkpoint"]),
        ),
        model_factory=tiny_3d_video_model,
        dataset_factory=SyntheticNormalClips,
    ).execute()
    resumed_checkpoint = torch.load(
        resumed["paths"]["last_checkpoint"], weights_only=True
    )
    assert resumed_checkpoint["completed_epoch"] == 2
    assert torch.equal(resumed_checkpoint["svdd_center"], checkpoint["svdd_center"])


def test_metrics_threshold_and_frame_aggregation() -> None:
    metrics, _, matrix = compute_binary_metrics(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5
    )
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert matrix.tolist() == [[2, 0], [0, 2]]

    frames = aggregate_frame_scores(
        [
            {"source": "a", "frame_indices": [1, 2], "ground_truth": 0, "anomaly_score": 0.2, "threshold": 0.5},
            {"source": "a", "frame_indices": [2, 3], "ground_truth": 1, "anomaly_score": 0.8, "threshold": 0.5},
        ],
        method="mean",
    )
    assert frames[1]["frame"] == 2
    assert frames[1]["anomaly_score"] == pytest.approx(0.5)
    assert frames[1]["ground_truth"] == 1


class SyntheticBenchmarkClips(Dataset):
    def __init__(self, *_args, **_kwargs):
        values = [0.1, 0.2, 0.8, 0.9]
        self.items = [
            (
                torch.full((2, 3, 2, 2), value),
                torch.tensor(int(value > 0.5)),
                {
                    "source": f"{'anomaly' if value > 0.5 else 'normal'}/clip-{index}",
                    "start_frame": index * 2,
                    "end_frame": index * 2 + 1,
                    "frame_indices": [index * 2, index * 2 + 1],
                },
            )
            for index, value in enumerate(values)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class MeanScoreModel(nn.Module):
    def forward(self, clips):
        return SimpleNamespace(anomaly_score=clips.mean(dim=(1, 2, 3, 4)))


def _fake_checkpoint_loader(_path, **_kwargs):
    checkpoint = {
        "model_name": "resnet18",
        "temporal_model": "tcn",
        "svdd_hidden_dim": 4,
        "svdd_dim": 2,
        "svdd_quantile": 0.75,
        "svdd_threshold": 0.5,
        "score_type": "squared_euclidean",
        "clip_length": 2,
        "frame_stride": 1,
        "input_height": 2,
        "input_width": 2,
    }
    stored = {
        "clip_length": 2,
        "frame_stride": 1,
        "height": 2,
        "width": 2,
    }
    return MeanScoreModel(), checkpoint, stored


def test_benchmark_uses_stored_threshold_and_writes_research_artifacts(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"stable fake checkpoint")
    output = tmp_path / "benchmark"
    result = BenchmarkVideoAnomalyModel(
        BenchmarkVideoAnomalyRequest(
            model_path=str(checkpoint_path),
            dataset_path="unused",
            output_path=str(output),
            batch_size=2,
            workers=0,
            plots=False,
        ),
        checkpoint_loader=_fake_checkpoint_loader,
        dataset_factory=SyntheticBenchmarkClips,
    ).execute()

    assert result["metrics"]["clip_level"]["auroc"] == pytest.approx(1.0)
    assert {row["threshold"] for row in result["predictions"]} == {0.5}
    for name in (
        "metrics.json",
        "metrics.csv",
        "run_metadata.json",
        "predictions.csv",
        "predictions.jsonl",
        "frame_predictions.csv",
        "roc_curve.csv",
        "pr_curve.csv",
        "benchmark_report.md",
    ):
        assert (output / name).is_file()


class FakeFrameSource:
    def __init__(self, values=range(4), *, fps: float | None = 2.0):
        self.frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in values]
        self.fps = fps
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True

    def metadata(self):
        return FrameSourceMetadata(fps=self.fps, frame_count=len(self.frames))


class FakeFrameSink:
    def __init__(self, *, stop_after: int | None = None):
        self.frames = []
        self.stop_after = stop_after
        self.closed = False

    def show(self, frame):
        self.frames.append(frame)
        return self.stop_after is None or len(self.frames) < self.stop_after

    def close(self):
        self.closed = True


def test_headless_video_inference_reports_normal_video_and_writes_artifacts(tmp_path) -> None:
    source = FakeFrameSource()
    output = tmp_path / "inference"
    events = []
    result = InferVideoAnomaly(
        InferVideoAnomalyRequest(
            model_path="unused.pth",
            file_path="unused.mp4",
            output_path=str(output),
            batch_size=2,
        ),
        checkpoint_loader=_fake_checkpoint_loader,
        frame_source=source,
        frame_transform=lambda frame: torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255,
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    assert [(row["start_frame"], row["end_frame"]) for row in result.predictions] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert result.anomaly_detected is False
    assert result.windows_scored == 3
    assert result.anomalous_windows == 0
    assert result.predictions[1]["start_time_seconds"] == pytest.approx(0.5)
    assert source.released is True
    assert (output / "predictions.jsonl").is_file()
    assert (output / "predictions.csv").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == result.summary()
    assert json_safe(result)["predictions"] == list(result.predictions)
    assert events[-1].level == "success"
    assert events[-1].message.startswith("No anomaly detected")


def test_headless_video_inference_reports_anomaly_when_any_window_exceeds_threshold(
    tmp_path,
) -> None:
    source = FakeFrameSource([0, 255, 255, 0], fps=None)
    events = []
    result = InferVideoAnomaly(
        InferVideoAnomalyRequest(
            model_path="unused.pth",
            file_path="sample.mp4",
            output_path=str(tmp_path / "inference"),
            batch_size=2,
        ),
        checkpoint_loader=_fake_checkpoint_loader,
        frame_source=source,
        frame_transform=lambda frame: torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255,
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    assert result.anomaly_detected is True
    assert result.anomalous_windows == 1
    assert result.max_anomaly_score == pytest.approx(1.0)
    assert [row["is_anomaly"] for row in result.predictions] == [False, True, False]
    assert all(row["start_time_seconds"] is None for row in result.predictions)
    assert events[-1].level == "warning"
    assert events[-1].message.startswith("Anomaly detected")


def test_displayed_video_inference_labels_every_frame_and_scores_complete_windows(
    tmp_path,
) -> None:
    source = FakeFrameSource([0, 255, 255, 0])
    sink = FakeFrameSink()
    rendered_predictions = []

    def renderer(frame, prediction, frames_buffered, frames_required):
        rendered_predictions.append(
            (
                None if prediction is None else prediction["is_anomaly"],
                frames_buffered,
                frames_required,
            )
        )
        return frame.copy()

    result = InferVideoAnomaly(
        InferVideoAnomalyRequest(
            model_path="unused.pth",
            file_path="sample.mp4",
            output_path=str(tmp_path / "inference"),
            batch_size=8,
        ),
        checkpoint_loader=_fake_checkpoint_loader,
        frame_source=source,
        frame_transform=lambda frame: torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255,
        frame_sink=sink,
        frame_renderer=renderer,
    ).execute()

    assert rendered_predictions == [
        (None, 1, 2),
        (False, 2, 2),
        (True, 2, 2),
        (False, 2, 2),
    ]
    assert result.frames_displayed == 4
    assert result.windows_scored == 3
    assert result.anomaly_detected is True
    assert result.stopped_by_user is False
    assert source.released is True
    assert sink.closed is True


def test_displayed_video_inference_stops_cleanly_during_warmup(tmp_path) -> None:
    source = FakeFrameSource([1, 2, 3])
    sink = FakeFrameSink(stop_after=1)
    result = InferVideoAnomaly(
        InferVideoAnomalyRequest(
            model_path="unused.pth",
            output_path=str(tmp_path),
        ),
        checkpoint_loader=_fake_checkpoint_loader,
        frame_source=source,
        frame_transform=lambda frame: torch.from_numpy(frame.transpose(2, 0, 1)).float(),
        frame_sink=sink,
        frame_renderer=lambda frame, *_args: frame,
    ).execute()

    assert result.stopped_by_user is True
    assert result.frames_displayed == 1
    assert result.windows_scored == 0
    assert result.max_anomaly_score is None
    assert source.released is True
    assert sink.closed is True


def test_video_anomaly_frame_annotation_is_colored_and_does_not_mutate_input() -> None:
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    warming = annotate_video_anomaly_frame(frame, None, 3, 16)
    normal = annotate_video_anomaly_frame(
        frame,
        {"is_anomaly": False, "anomaly_score": 0.2, "threshold": 0.5},
        16,
        16,
    )
    anomaly = annotate_video_anomaly_frame(
        frame,
        {"is_anomaly": True, "anomaly_score": 0.8, "threshold": 0.5},
        16,
        16,
    )

    assert np.count_nonzero(warming) > 0
    assert np.count_nonzero(normal[:, :, 1]) > np.count_nonzero(normal[:, :, 2])
    assert np.count_nonzero(anomaly[:, :, 2]) > np.count_nonzero(anomaly[:, :, 1])
    assert np.count_nonzero(frame) == 0


def test_video_anomaly_display_requires_sink_and_renderer_together() -> None:
    with pytest.raises(ValueError, match="both a frame sink and frame renderer"):
        InferVideoAnomaly(
            InferVideoAnomalyRequest(model_path="unused.pth"),
            frame_sink=FakeFrameSink(),
        )


def test_video_inference_uses_video_factory_and_releases_short_source(tmp_path) -> None:
    source = FakeFrameSource([1])
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return source

    with pytest.raises(MLXUserError, match="at least 2 decoded frames"):
        InferVideoAnomaly(
            InferVideoAnomalyRequest(
                model_path="unused.pth",
                file_path="sample.mp4",
                output_path=str(tmp_path),
            ),
            checkpoint_loader=_fake_checkpoint_loader,
            frame_source_factory=factory,
            frame_transform=lambda frame: torch.from_numpy(frame.transpose(2, 0, 1)).float(),
        ).execute()

    assert factory_calls == [{"source": "video", "file_path": "sample.mp4"}]
    assert source.released is True


def test_video_inference_rejects_nonfinite_scores_and_releases_source(tmp_path) -> None:
    class NonFiniteModel(nn.Module):
        def forward(self, clips):
            return SimpleNamespace(anomaly_score=torch.full((len(clips),), float("nan")))

    def loader(*_args, **_kwargs):
        model, checkpoint, stored = _fake_checkpoint_loader(None)
        return NonFiniteModel(), checkpoint, stored

    source = FakeFrameSource([1, 2])
    sink = FakeFrameSink()
    with pytest.raises(MLXUserError, match="non-finite anomaly score"):
        InferVideoAnomaly(
            InferVideoAnomalyRequest(
                model_path="unused.pth",
                output_path=str(tmp_path),
            ),
            checkpoint_loader=loader,
            frame_source=source,
            frame_transform=lambda frame: torch.from_numpy(frame.transpose(2, 0, 1)).float(),
            frame_sink=sink,
            frame_renderer=lambda frame, *_args: frame,
        ).execute()

    assert source.released is True
    assert sink.closed is True


@pytest.mark.parametrize("threshold", [None, "invalid", float("nan"), float("inf")])
def test_video_inference_requires_finite_calibrated_threshold(threshold) -> None:
    def loader(*_args, **_kwargs):
        model, checkpoint, stored = _fake_checkpoint_loader(None)
        checkpoint["svdd_threshold"] = threshold
        return model, checkpoint, stored

    with pytest.raises(MLXUserError, match="calibrated threshold"):
        InferVideoAnomaly(
            InferVideoAnomalyRequest(model_path="unused.pth"),
            checkpoint_loader=loader,
            frame_source=FakeFrameSource(),
        ).execute()


def test_video_inference_requires_positive_batch_size() -> None:
    with pytest.raises(MLXUserError, match="batch size must be a positive integer"):
        InferVideoAnomaly(
            InferVideoAnomalyRequest(model_path="unused.pth", batch_size=0),
            checkpoint_loader=_fake_checkpoint_loader,
            frame_source=FakeFrameSource(),
        ).execute()


def test_video_inference_releases_source_when_metadata_loading_fails() -> None:
    class BrokenMetadataSource(FakeFrameSource):
        def metadata(self):
            raise MLXUserError("bad metadata")

    source = BrokenMetadataSource()
    with pytest.raises(MLXUserError, match="bad metadata"):
        InferVideoAnomaly(
            InferVideoAnomalyRequest(model_path="unused.pth"),
            checkpoint_loader=_fake_checkpoint_loader,
            frame_source=source,
        ).execute()

    assert source.released is True


def test_cli_mode_aliases_and_options() -> None:
    target = "mlx.modes.video_anomaly_detection.runner:run_video_anomaly_detection"
    assert MODE_REGISTRY["video_anomaly_detection"] == target
    assert MODE_REGISTRY["video-anomaly-detection"] == target
    parsed = build_parser().parse_args(
        [
            "--clip-length",
            "8",
            "--frame-stride",
            "2",
            "--backbone-mode",
            "3d",
            "--backbone-temporal-kernel-size",
            "5",
        ]
    )
    assert (
        parsed.clip_length,
        parsed.frame_stride,
        parsed.backbone_mode,
        parsed.backbone_temporal_kernel_size,
    ) == (8, 2, "3d", 5)


def test_model_factory_defaults_to_3d_and_retains_legacy_path() -> None:
    three_d = build_video_anomaly_model(
        "resnet18",
        {
            "pretrained": False,
            "clip_length": 3,
            "svdd_dim": 2,
            "svdd_hidden_dim": 4,
        },
        backbone_3d_factory=lambda *_args: Tiny3DBackbone(),
    )
    legacy = build_video_anomaly_model(
        "resnet18",
        {
            "backbone_mode": "frame-2d",
            "temporal_hidden_dim": 5,
            "temporal_embedding_dim": 3,
            "svdd_dim": 2,
            "svdd_hidden_dim": 4,
        },
        backbone_factory=lambda *_args: TinyFeatureBackbone(),
    )
    assert isinstance(three_d, VideoAnomaly3DModel)
    assert isinstance(legacy, VideoAnomalyModel)


def test_checkpoint_v2_metadata_and_v1_legacy_reconstruction() -> None:
    model = tiny_3d_video_model()
    config = {
        "model": "resnet18",
        "backbone_mode": "3d",
        "clip_length": 3,
        "frame_stride": 1,
        "height": 8,
        "width": 8,
        "svdd_dim": 2,
        "svdd_hidden_dim": 5,
        "svdd_quantile": 0.95,
    }
    metadata = model_metadata(model, config)
    assert metadata["checkpoint_version"] == 2
    assert metadata["backbone_mode"] == "3d"
    assert metadata["backbone_temporal_stride_policy"] == "preserve"
    assert artifact_stem(config) == "resnet18-3d-svdd"

    legacy = {
        "model_name": "resnet18",
        "input_height": 8,
        "input_width": 8,
        "clip_length": 3,
        "frame_stride": 1,
        "temporal_model": "tcn",
        "temporal_hidden_dim": 5,
        "temporal_embedding_dim": 3,
        "temporal_kernel_size": 3,
        "temporal_dropout": 0.0,
        "svdd_dim": 2,
        "svdd_hidden_dim": 4,
        "svdd_quantile": 0.95,
    }
    assert config_from_checkpoint(legacy)["backbone_mode"] == "frame-2d"


def test_versioned_checkpoint_loader_reconstructs_3d_and_legacy(tmp_path) -> None:
    three_d_path = tmp_path / "three-d.pth"
    three_d = tiny_3d_video_model()
    three_d.svdd_head.center.copy_(torch.tensor([0.1, -0.2]))
    three_d.svdd_head.threshold.copy_(torch.tensor(0.7))
    save_deployment_checkpoint(
        three_d_path,
        three_d,
        {
            "model": "resnet18",
            "backbone_mode": "3d",
            "backbone_temporal_kernel_size": 3,
            "height": 8,
            "width": 8,
            "clip_length": 3,
            "frame_stride": 1,
            "svdd_dim": 2,
            "svdd_hidden_dim": 5,
            "svdd_quantile": 0.95,
        },
    )
    loaded, checkpoint, stored = load_video_anomaly_checkpoint(
        three_d_path,
        model_factory=tiny_3d_video_model,
    )
    assert isinstance(loaded, VideoAnomaly3DModel)
    assert checkpoint["checkpoint_version"] == 2
    assert stored["backbone_mode"] == "3d"

    legacy_path = tmp_path / "legacy.pth"
    legacy_model = tiny_video_model()
    legacy_model.svdd_head.center.copy_(torch.tensor([0.2, -0.1]))
    legacy_model.svdd_head.threshold.copy_(torch.tensor(0.6))
    legacy_config = {
        "model": "resnet18",
        "backbone_mode": "frame-2d",
        "height": 8,
        "width": 8,
        "clip_length": 3,
        "frame_stride": 1,
        "temporal_model": "tcn",
        "temporal_hidden_dim": 5,
        "temporal_embedding_dim": 3,
        "temporal_kernel_size": 3,
        "temporal_dropout": 0.0,
        "svdd_dim": 2,
        "svdd_hidden_dim": 4,
        "svdd_quantile": 0.95,
    }
    legacy_payload = model_metadata(legacy_model, legacy_config)
    legacy_payload.pop("backbone_mode")
    legacy_payload["checkpoint_version"] = 1
    legacy_payload["state_dict"] = legacy_model.state_dict()
    torch.save(legacy_payload, legacy_path)
    _, _, legacy_stored = load_video_anomaly_checkpoint(
        legacy_path,
        model_factory=tiny_video_model,
    )
    assert legacy_stored["backbone_mode"] == "frame-2d"


def test_runner_3d_default_legacy_selection_and_conflict(monkeypatch) -> None:
    from mlx.modes.video_anomaly_detection import runner

    captured = []
    monkeypatch.setitem(runner.ACTION_HANDLERS, "train", lambda config: captured.append(config))
    runner.run_video_anomaly_detection({"action": "train"})
    assert captured[-1]["backbone_mode"] == "3d"

    runner.run_video_anomaly_detection(
        {"action": "train", "_explicit_options": {"temporal_model"}}
    )
    assert captured[-1]["backbone_mode"] == "frame-2d"

    with pytest.raises(MLXUserError, match="apply only"):
        runner.run_video_anomaly_detection(
            {
                "action": "train",
                "backbone_mode": "3d",
                "_explicit_options": {"backbone_mode", "temporal_model"},
            }
        )


def test_inference_runner_uses_checkpoint_model_unless_model_is_explicit(
    monkeypatch,
) -> None:
    from mlx.modes.video_anomaly_detection import runner

    requests = []
    inference_options = []
    sink_options = []
    sink = FakeFrameSink()

    class FakeOpenCVFrameSink:
        def __new__(cls, **kwargs):
            sink_options.append(kwargs)
            return sink

    class FakeInference:
        def __init__(self, request, **kwargs):
            requests.append(request)
            inference_options.append(kwargs)

        def execute(self):
            return "ok"

    monkeypatch.setattr(runner, "InferVideoAnomaly", FakeInference)
    monkeypatch.setattr(runner, "OpenCVFrameSink", FakeOpenCVFrameSink)

    assert runner.run_video_anomaly_detection({"action": "infer-video"}) == "ok"
    assert requests[-1].model is None
    assert sink_options[-1] == {
        "title": "MLX Video Anomaly Detection",
        "delay_ms": 10,
    }
    assert inference_options[-1]["frame_sink"] is sink
    assert inference_options[-1]["frame_renderer"] is annotate_video_anomaly_frame

    assert (
        runner.run_video_anomaly_detection(
            {
                "action": "infer-video",
                "model": "densenet121",
                "_explicit_options": {"model"},
            }
        )
        == "ok"
    )
    assert requests[-1].model == "densenet121"

    sink_options.clear()
    assert (
        runner.run_video_anomaly_detection(
            {"action": "infer-video", "display": False}
        )
        == "ok"
    )
    assert requests[-1].model is None
    assert inference_options[-1]["frame_sink"] is None
    assert inference_options[-1]["frame_renderer"] is None
    assert sink_options == []

    assert (
        runner.run_video_anomaly_detection(
            {"action": "infer-video", "output_format": "json"}
        )
        == "ok"
    )
    assert inference_options[-1]["frame_sink"] is None
    assert inference_options[-1]["frame_renderer"] is None
    assert sink_options == []


def test_train_composition_normalizes_cli_explicitness_into_typed_request(
    monkeypatch, tmp_path
) -> None:
    from mlx.modes.video_anomaly_detection import runner

    captured = {}

    class FakeDatasetSourceTraining:
        def __init__(self, request, **_kwargs):
            captured["request"] = request

        def execute(self):
            return "ok"

    monkeypatch.setattr(runner, "TrainWithDatasetSource", FakeDatasetSourceTraining)
    result = runner._train(
        {
            "model": "resnet18",
            "output_path": str(tmp_path),
            "_explicit_options": {"backbone_mode", "temporal_dropout"},
        }
    )

    assert result == "ok"
    assert captured["request"].backbone_mode_explicit is True
    assert captured["request"].temporal_options_explicit is True


def test_train_composition_removes_implicit_local_dataset_for_s3(
    monkeypatch, tmp_path
) -> None:
    from mlx.modes.video_anomaly_detection import runner

    captured = {}

    class FakeDatasetSourceTraining:
        def __init__(self, request, **_kwargs):
            captured["request"] = request

        def execute(self):
            return "ok"

    monkeypatch.setattr(runner, "TrainWithDatasetSource", FakeDatasetSourceTraining)
    result = runner._train(
        {
            "model": "resnet18",
            "dataset_path": "./tmp/dataset",
            "dataset_s3_uri": "s3://datasets/avenue.zip",
            "output_path": str(tmp_path),
            "_explicit_options": {"dataset_s3_uri"},
        }
    )

    assert result == "ok"
    assert captured["request"].dataset_path == ""
    assert captured["request"].dataset_s3_uri == "s3://datasets/avenue.zip"


def test_video_model_registries_are_read_only_and_extensible() -> None:
    from mlx.modes.video_anomaly_detection.models import TEMPORAL_ENCODERS
    from mlx.modes.video_anomaly_detection.models.temporal import (
        DEFAULT_TEMPORAL_ENCODER_REGISTRY,
    )
    builder = lambda *args, **kwargs: torch.nn.Identity()

    with pytest.raises(TypeError):
        TEMPORAL_ENCODERS["custom"] = builder

    extended = DEFAULT_TEMPORAL_ENCODER_REGISTRY.register("custom", builder)
    assert "custom" in extended.entries
    assert "custom" not in TEMPORAL_ENCODERS
