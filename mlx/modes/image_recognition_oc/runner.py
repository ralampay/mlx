from __future__ import annotations

from typing import Any

from mlx.core.commands import NullWorkflowReporter
from mlx.core.datasets import TrainWithDatasetSource, validate_dataset_source_options
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.artifacts import resolve_training_paths
from mlx.modes.image_recognition_oc.commands import (
    BenchmarkImageOneClass,
    InferImageOneClass,
    ListImageOneClassModels,
    TrainImageOneClassModel,
)
from mlx.modes.image_recognition_oc.data import image_one_class_dataset_root
from mlx.modes.image_recognition_oc.presentation import (
    RichImageOneClassReporter,
    display_inference,
    display_models,
)
from mlx.modes.image_recognition_oc.requests import (
    BenchmarkImageOneClassRequest,
    InferImageOneClassRequest,
    ListImageOneClassModelsRequest,
    TrainImageOneClassRequest,
)


DEFAULT_CONFIG = {
    "action": "ls-models",
    "model": "deep-svdd",
    "backbone": "resnet18",
    "height": 224,
    "width": 224,
    "batch_size": 16,
    "workers": 0,
    "epochs": 50,
    "lr": 0.001,
    "colored": True,
    "pretrained": False,
    "random_seed": None,
    "drax_fusion_mode": "average",
    "svdd_dim": 128,
    "svdd_hidden_dim": 256,
    "svdd_quantile": 0.95,
    "use_best": True,
    "apply_transformations": False,
}


def _reporter(config: dict[str, Any]):
    return NullWorkflowReporter() if config.get("output_format") == "json" else RichImageOneClassReporter()


def _train(config: dict[str, Any]):
    request_config = dict(config)
    explicit = set(config.get("_explicit_options") or ())
    if request_config.get("dataset_s3_uri") and "dataset_path" not in explicit:
        request_config["dataset_path"] = ""
    request = TrainImageOneClassRequest.from_config(request_config)
    reporter = _reporter(config)
    return TrainWithDatasetSource(
        request,
        trainer_factory=lambda resolved: TrainImageOneClassModel(resolved, reporter=reporter),
        root_resolver=image_one_class_dataset_root,
        artifact_dir_resolver=lambda resolved: resolve_training_paths(resolved.to_config())["output_dir"],
        profile=config.get("profile"),
        reporter=reporter,
    ).execute()


def _infer(config: dict[str, Any]):
    explicit = set(config.get("_explicit_options") or ())
    request_config = dict(config)
    if "model" not in explicit:
        request_config["model"] = None
    if "backbone" not in explicit:
        request_config["backbone"] = None
    if "input_img" not in explicit:
        request_config["input_img"] = None
    result = InferImageOneClass(
        InferImageOneClassRequest.from_config(request_config),
        reporter=_reporter(config),
    ).execute()
    if config.get("output_format") != "json":
        display_inference(result)
    return result


def _benchmark(config: dict[str, Any]):
    explicit = set(config.get("_explicit_options") or ())
    request_config = dict(config)
    if "model" not in explicit:
        request_config["model"] = None
    if "backbone" not in explicit:
        request_config["backbone"] = None
    return BenchmarkImageOneClass(
        BenchmarkImageOneClassRequest.from_config(request_config),
        reporter=_reporter(config),
    ).execute()


def _list_models(config: dict[str, Any]):
    explicit = set(config.get("_explicit_options") or ())
    request_config = dict(config)
    if "model" not in explicit:
        request_config["model"] = None
    if "backbone" not in explicit:
        request_config["backbone"] = None
    result = ListImageOneClassModels(
        ListImageOneClassModelsRequest.from_config(request_config),
        reporter=_reporter(config),
    ).execute()
    if config.get("output_format") != "json":
        display_models(result)
    return result


ACTION_HANDLERS = {
    "benchmark": _benchmark,
    "infer-image": _infer,
    "ls-models": _list_models,
    "train": _train,
}


def run_image_recognition_oc(mode_config: dict[str, Any]):
    if mode_config.get("platform", "local") == "aws":
        from mlx.modes.image_recognition_oc.aws.runner import run_aws_image_one_class

        return run_aws_image_one_class(mode_config)
    explicit = set(mode_config.get("_explicit_options") or ())
    config = dict(mode_config)
    for name, default in DEFAULT_CONFIG.items():
        if name not in explicit:
            config[name] = default
    action = config.get("action") or "ls-models"
    config["action"] = action
    validate_dataset_source_options(config, action=action)
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for image-recognition-oc. Available actions: {available}."
        )
    return handler(config)


__all__ = ["ACTION_HANDLERS", "DEFAULT_CONFIG", "run_image_recognition_oc"]
