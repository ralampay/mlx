from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx.core.datasets import (
    TrainWithDatasetSource,
    validate_dataset_source_options,
)
from mlx.core.commands import NullWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.configuration import with_explicit_options, normalize_image_dimensions
from mlx.core.extensions import deferred_callable
from mlx.core.ui import print_model_parameter_table
from mlx.modes.image_classification.models import DEFAULT_MODEL, model_family_for
from mlx.modes.image_classification.requests import (
    BuildImageClassificationDatasetRequest,
    BenchmarkImageClassificationRequest,
    GenerateImageClassificationCamsRequest,
    ImageClassificationRequest,
    InferImageClassificationRequest,
    SmokeTestImageClassificationRequest,
    TrainImageClassificationRequest,
)


GenerateImageClassificationCams = deferred_callable(
    "mlx.modes.image_classification.cam:GenerateImageClassificationCams"
)
BuildImageClassificationDataset = deferred_callable(
    "mlx.modes.image_classification.data:BuildImageClassificationDataset"
)
classification_dataset_root = deferred_callable(
    "mlx.modes.image_classification.data:classification_dataset_root"
)
BenchmarkImageClassification = deferred_callable(
    "mlx.modes.image_classification.evaluation:BenchmarkImageClassification"
)
InferImageClassification = deferred_callable(
    "mlx.modes.image_classification.inference:InferImageClassification"
)
ListImageClassificationModels = deferred_callable(
    "mlx.modes.image_classification.list_models:ListImageClassificationModels"
)
display_cam_results = deferred_callable(
    "mlx.modes.image_classification.presentation:display_cam_results"
)
display_classification_predictions = deferred_callable(
    "mlx.modes.image_classification.presentation:display_classification_predictions"
)
display_similarity_matches = deferred_callable(
    "mlx.modes.image_classification.presentation:display_similarity_matches"
)
print_config_summary = deferred_callable(
    "mlx.modes.image_classification.presentation:print_config_summary"
)
RichImageClassificationReporter = deferred_callable(
    "mlx.modes.image_classification.presentation:RichImageClassificationReporter"
)
resolve_image_dataset_build_request = deferred_callable(
    "mlx.modes.image_classification.presentation:resolve_image_dataset_build_request"
)
SmokeTestImageClassificationModel = deferred_callable(
    "mlx.modes.image_classification.train:SmokeTestImageClassificationModel"
)
TrainImageClassificationModel = deferred_callable(
    "mlx.modes.image_classification.train:TrainImageClassificationModel"
)

DEFAULT_CONFIG = {
    "action": "test",
    "apply_transformations": False,
    "batch_size": 1,
    "colored": True,
    "dataset_path": "",
    "device": "cpu",
    "embedding_size": 4096,
    "epochs": 100,
    "input_size": (256, 256),
    "lr": None,
    "num_pairs": 100,
    "ood_method": "none",
    "pretrained": False,
    "random_seed": None,
    "refresh_per_second": 2,
    "use_best": True,
    "verbose": False,
    "svdd_weight": 0.05,
    "svdd_dim": 128,
    "svdd_hidden_dim": 256,
    "svdd_quantile": 0.95,
    "svdd_warmup_epochs": 0,
}


def _reporter(config: dict[str, Any]):
    return NullWorkflowReporter() if config.get("output_format") == "json" else RichImageClassificationReporter()


def _list_models(config: dict[str, Any]):
    if config.get("names_only"):
        from mlx.modes.image_classification.models import supported_model_names
        from mlx.core.model_listing import ListComponentNames
        from mlx.core.presentation import display_component_inventory
        result = ListComponentNames(supported_model_names()).execute()
        if config.get("output_format") != "json":
            display_component_inventory(result, title="Available Models")
        return result

    summaries = ListImageClassificationModels(config).execute()
    if config.get("output_format") != "json":
        print_model_parameter_table(summaries, title="Image Classification Models")
    return summaries


def _infer_image(config: dict[str, Any]):
    result = InferImageClassification(
        InferImageClassificationRequest.from_config(config),
        reporter=_reporter(config),
    ).execute()
    if config.get("output_format") != "json" and config.get("display", True):
        if "top_matches" in result:
            display_similarity_matches(result)
        else:
            display_classification_predictions(result)
    return result


def _generate_cams(config: dict[str, Any]):
    results = GenerateImageClassificationCams(
        GenerateImageClassificationCamsRequest.from_config(config),
        reporter=_reporter(config),
    ).execute()
    if config.get("output_format") != "json" and config.get("display", True):
        display_cam_results(results, delay=int(config.get("window_delay", 0)))
    return results


def _train(config: dict[str, Any]):
    from mlx.core.configuration import load_component_options
    config = {**config, "loss_config": load_component_options(config.get("loss_config"), purpose="loss")}
    request = TrainImageClassificationRequest.from_config(config)
    reporter = _reporter(config)
    return TrainWithDatasetSource(
        request,
        trainer_factory=lambda resolved: TrainImageClassificationModel(
            resolved, reporter=reporter
        ),
        root_resolver=classification_dataset_root,
        artifact_dir_resolver=lambda resolved: Path(str(resolved.output_path)),
        profile=config.get("profile"),
        reporter=reporter,
    ).execute()


def _list_losses(config):
    from mlx.core.model_listing import ListComponentNames
    from mlx.core.presentation import display_component_inventory
    from mlx.modes.image_classification.losses import LOSS_DEFINITIONS
    result = ListComponentNames(LOSS_DEFINITIONS).execute()
    if config.get("output_format") != "json":
        display_component_inventory(result, title="Available Losses")
    return result


def _list_cam_methods(config):
    from mlx.core.model_listing import ListComponentNames
    from mlx.core.presentation import display_component_inventory
    from mlx.modes.image_classification.cam_registry import DEFAULT_CAM_REGISTRY
    result = ListComponentNames(DEFAULT_CAM_REGISTRY.entries).execute()
    if config.get("output_format") != "json":
        display_component_inventory(result, title="CAM Methods")
    return result


ACTION_HANDLERS = {
    "ls-cam-methods": _list_cam_methods,
    "ls-losses": _list_losses,
    "benchmark": lambda config: BenchmarkImageClassification(
        BenchmarkImageClassificationRequest.from_config(config),
        reporter=_reporter(config),
    ).execute(),
    "build-dataset": lambda config: BuildImageClassificationDataset(
        BuildImageClassificationDatasetRequest.from_config(config),
        reporter=_reporter(config),
        input_resolver=resolve_image_dataset_build_request,
    ).execute(),
    "infer-image": _infer_image,
    "ls-models": _list_models,
    "cam": _generate_cams,
    "test": lambda config: SmokeTestImageClassificationModel(
        SmokeTestImageClassificationRequest.from_config(config),
        reporter=_reporter(config),
    ).execute(),
    "train": _train,
}


def run_image_classification(mode_config: dict[str, Any]) -> Any:
    if mode_config.get("platform", "local") == "aws":
        from mlx.modes.image_classification.aws.runner import run_aws_image_classification

        return run_aws_image_classification(mode_config)

    config = {**DEFAULT_CONFIG, **mode_config}
    explicit = with_explicit_options(mode_config)["_explicit_options"]
    if "workers" not in explicit:
        config["workers"] = 2
    action = config.get("action") or DEFAULT_CONFIG["action"]
    config["action"] = action
    validate_dataset_source_options(config, action=action)
    if action in {"ls-models", "ls-losses", "ls-cam-methods"}:
        return ACTION_HANDLERS[action](config)

    model_name = mode_config.get("model") or DEFAULT_MODEL
    family = model_family_for(model_name)
    config["_explicit_options"] = explicit
    default_size = (105, 105) if family == "one-shot" and mode_config.get("model") is None else (224, 224)
    # Explicit model selection historically uses the shared 256-pixel CLI default.
    if mode_config.get("model") is not None:
        default_size = (256, 256)
    config = normalize_image_dimensions(config, default=default_size)
    config["model"] = model_name

    if config.get("output_format") != "json":
        print_config_summary(model_name, family, config)

    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for image-classification. Available actions: {available}."
        )

    return handler(config)
