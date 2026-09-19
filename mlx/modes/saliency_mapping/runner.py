from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.commands import NullWorkflowReporter
from mlx.core.datasets import TrainWithDatasetSource, validate_dataset_source_options
from mlx.core.exceptions import MLXUserError
from mlx.modes.saliency_mapping.benchmark import (
    BenchmarkSaliencyMapping,
    BenchmarkSaliencyModelGroup,
)
from mlx.modes.saliency_mapping.checkpoints import training_paths
from mlx.modes.saliency_mapping.commands import TrainSaliencyModelGroup
from mlx.modes.saliency_mapping.data import BuildSaliencyDataset, saliency_dataset_root
from mlx.modes.saliency_mapping.inference import InferSaliencyImage
from mlx.modes.saliency_mapping.list_models import ListSaliencyModels
from mlx.modes.saliency_mapping.models import DEFAULT_MODEL, MODEL_GROUP_NAMES
from mlx.modes.saliency_mapping.presentation import (
    RichSaliencyReporter,
    print_config,
    print_model_table,
    resolve_dataset_build_request,
)
from mlx.modes.saliency_mapping.requests import (
    BenchmarkSaliencyRequest,
    BuildSaliencyDatasetRequest,
    SaliencyRequest,
    TrainSaliencyRequest,
)
from mlx.modes.saliency_mapping.train import SmokeTestSaliencyModels, TrainSaliencyModel
from mlx.modes.segmentation.data import normalize_segmentation_transform

DEFAULT_CONFIG = {
    "action": "test",
    "batch_size": 4,
    "bce_weight": 1.0,
    "colored": True,
    "dataset_path": "",
    "device": "cpu",
    "epochs": 50,
    "input_size": (256, 256),
    "iou_weight": 1.0,
    "lr": None,
    "mask_threshold": 0.5,
    "overlay_alpha": 0.45,
    "split": "test",
    "ssim_weight": 1.0,
    "threshold_steps": 101,
    "transform": "resize",
    "workers": 0,
}


def _reporter(config):
    return NullWorkflowReporter() if config.get("output_format") == "json" else RichSaliencyReporter()


def _train(config):
    request_config = dict(config)
    explicit = set(config.get("_explicit_options") or ())
    if request_config.get("dataset_s3_uri") and "dataset_path" not in explicit:
        request_config["dataset_path"] = ""
    request = TrainSaliencyRequest.from_config(request_config)
    reporter = _reporter(config)
    grouped = request.model in MODEL_GROUP_NAMES

    def factory(resolved):
        if grouped:
            return TrainSaliencyModelGroup(resolved, reporter=reporter)
        return TrainSaliencyModel(resolved, reporter=reporter)

    def artifacts(resolved):
        if grouped:
            if not resolved.output_path:
                raise MLXUserError("Grouped saliency training requires --output.")
            return Path(resolved.output_path).expanduser()
        return training_paths(resolved.to_config(), model_name=str(resolved.model))["output_dir"]

    return TrainWithDatasetSource(
        request,
        trainer_factory=factory,
        root_resolver=saliency_dataset_root,
        artifact_dir_resolver=artifacts,
        profile=config.get("profile"),
        reporter=reporter,
    ).execute()


def _benchmark(config):
    request = BenchmarkSaliencyRequest.from_config(config)
    if request.model in MODEL_GROUP_NAMES:
        return BenchmarkSaliencyModelGroup(request, reporter=_reporter(config)).execute()
    return BenchmarkSaliencyMapping(request, reporter=_reporter(config)).execute()


def _list_models(config):
    result = ListSaliencyModels(config).execute()
    if config.get("output_format") != "json":
        print_model_table(result)
    return result


ACTION_HANDLERS = {
    "benchmark": _benchmark,
    "build-dataset": lambda config: BuildSaliencyDataset(
        BuildSaliencyDatasetRequest.from_config(config),
        reporter=_reporter(config),
        input_resolver=None if config.get("output_format") == "json" else resolve_dataset_build_request,
    ).execute(),
    "infer-image": lambda config: InferSaliencyImage(SaliencyRequest.from_config(config)).execute(),
    "ls-models": _list_models,
    "test": lambda config: SmokeTestSaliencyModels(
        SaliencyRequest.from_config(config), reporter=_reporter(config)
    ).execute(),
    "train": _train,
}


def run_saliency_mapping(mode_config: dict[str, Any]) -> Any:
    config = {**DEFAULT_CONFIG, **mode_config}
    config["action"] = config.get("action") or "test"
    validate_dataset_source_options(config, action=config["action"])
    if config["action"] == "ls-models":
        return _list_models(config)
    supplied_model = mode_config.get("model")
    if config["action"] in {"train", "test"}:
        config["model"] = supplied_model or DEFAULT_MODEL
    elif config["action"] == "benchmark":
        config["model"] = supplied_model or (None if config.get("model_path") else DEFAULT_MODEL)
    else:
        config["model"] = supplied_model
    config["input_size"] = tuple(config.get("input_size", (config["width"], config["height"])))
    config["transform"] = normalize_segmentation_transform(config.get("transform"))
    if config.get("output_format") != "json":
        print_config(config.get("model"), config)
    handler = ACTION_HANDLERS.get(config["action"])
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{config['action']}' for saliency mapping. Available actions: {available}."
        )
    return handler(config)


__all__ = ["ACTION_HANDLERS", "run_saliency_mapping"]
