from __future__ import annotations

from typing import Any

from mlx.core.datasets import (
    TrainWithDatasetSource,
    validate_dataset_source_options,
)
from mlx.core.commands import NullWorkflowReporter
from mlx.core.streaming import OpenCVFrameSink
from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.commands import (
    BenchmarkVideoAnomalyModel,
    InferVideoAnomaly,
    ListVideoAnomalyModels,
    TrainVideoAnomalyModel,
)
from mlx.modes.video_anomaly_detection.presentation import (
    RichVideoAnomalyReporter,
    annotate_video_anomaly_frame,
    display_video_anomaly_models,
)
from mlx.modes.video_anomaly_detection.requests import (
    BenchmarkVideoAnomalyRequest,
    InferVideoAnomalyRequest,
    ListVideoAnomalyModelsRequest,
    TrainVideoAnomalyRequest,
)
from mlx.modes.video_anomaly_detection.artifacts import resolve_training_paths
from mlx.modes.video_anomaly_detection.data import video_anomaly_dataset_root


DEFAULT_CONFIG = {
    "action": "ls-models",
    "model": "resnet18",
    "height": 224,
    "width": 224,
    "batch_size": 8,
    "workers": 0,
    "epochs": 50,
    "lr": 0.001,
    "clip_length": 16,
    "frame_stride": 1,
    "backbone_mode": "3d",
    "backbone_temporal_kernel_size": 3,
    "temporal_model": "tcn",
    "temporal_hidden_dim": 256,
    "temporal_embedding_dim": 128,
    "temporal_kernel_size": 3,
    "temporal_dropout": 0.0,
    "svdd_dim": 128,
    "svdd_hidden_dim": 256,
    "svdd_quantile": 0.95,
    "frame_aggregation": "mean",
}


def _reporter(config: dict[str, Any]):
    return NullWorkflowReporter() if config.get("output_format") == "json" else RichVideoAnomalyReporter()


def _list_models(config: dict[str, Any]):
    result = ListVideoAnomalyModels(
        ListVideoAnomalyModelsRequest.from_config(config),
        reporter=_reporter(config),
    ).execute()
    if config.get("output_format") != "json":
        display_video_anomaly_models(result)
    return result


def _train(config: dict[str, Any]):
    explicit = set(config.get("_explicit_options") or ())
    request_config = dict(config)
    if request_config.get("dataset_s3_uri") and "dataset_path" not in explicit:
        request_config["dataset_path"] = ""
    request = TrainVideoAnomalyRequest.from_config(
        {
            **request_config,
            "backbone_mode_explicit": "backbone_mode" in explicit,
            "temporal_options_explicit": bool(
                explicit
                & {
                    "temporal_model",
                    "temporal_hidden_dim",
                    "temporal_embedding_dim",
                    "temporal_kernel_size",
                    "temporal_dropout",
                }
            ),
        }
    )
    reporter = _reporter(config)
    return TrainWithDatasetSource(
        request,
        trainer_factory=lambda resolved: TrainVideoAnomalyModel(
            resolved, reporter=reporter
        ),
        root_resolver=video_anomaly_dataset_root,
        artifact_dir_resolver=lambda resolved: resolve_training_paths(
            resolved.to_config()
        )["output_dir"],
        profile=config.get("profile"),
        reporter=reporter,
    ).execute()


def _infer_video(config: dict[str, Any]):
    request_config = dict(config)
    if "model" not in set(config.get("_explicit_options") or ()):
        request_config["model"] = None
    display = bool(config.get("display", True)) and config.get("output_format") != "json"
    frame_sink = (
        OpenCVFrameSink(
            title="MLX Video Anomaly Detection",
            delay_ms=max(int(config.get("window_delay") or 10), 1),
        )
        if display
        else None
    )
    return InferVideoAnomaly(
        InferVideoAnomalyRequest.from_config(request_config),
        reporter=_reporter(config),
        frame_sink=frame_sink,
        frame_renderer=annotate_video_anomaly_frame if display else None,
    ).execute()


ACTION_HANDLERS = {
    "benchmark": lambda config: BenchmarkVideoAnomalyModel(
        BenchmarkVideoAnomalyRequest.from_config(config),
        reporter=_reporter(config),
    ).execute(),
    "infer-video": _infer_video,
    "ls-models": _list_models,
    "train": _train,
}


def run_video_anomaly_detection(mode_config: dict[str, Any]):
    config = {**DEFAULT_CONFIG, **mode_config}
    explicit = set(mode_config.get("_explicit_options") or ())
    legacy_temporal_options = {
        "temporal_model",
        "temporal_hidden_dim",
        "temporal_embedding_dim",
        "temporal_kernel_size",
        "temporal_dropout",
    }
    explicitly_legacy = bool(explicit & legacy_temporal_options)
    if "backbone_mode" not in explicit and explicitly_legacy:
        config["backbone_mode"] = "frame-2d"
    elif config.get("backbone_mode") == "3d" and explicitly_legacy:
        raise MLXUserError(
            "Temporal-CNN options apply only to --backbone-mode frame-2d; "
            "3D backbones learn space and time jointly."
        )
    for name, default in (("height", 224), ("width", 224), ("batch_size", 8), ("epochs", 50), ("lr", 0.001)):
        if name not in explicit:
            config[name] = default
    action = config.get("action") or "ls-models"
    validate_dataset_source_options(config, action=action)
    if action == "train" and not config.get("model"):
        config["model"] = "resnet18"
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for video-anomaly-detection. Available actions: {available}."
        )
    return handler(config)


__all__ = ["ACTION_HANDLERS", "DEFAULT_CONFIG", "run_video_anomaly_detection"]
