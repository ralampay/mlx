from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.image_classification.cam import GenerateImageClassificationCams
from mlx.modes.image_classification.data import BuildImageClassificationDataset
from mlx.modes.image_classification.evaluation import BenchmarkImageClassification
from mlx.modes.image_classification.inference import InferImageClassification
from mlx.modes.image_classification.list_models import ListImageClassificationModels
from mlx.modes.image_classification.models import DEFAULT_MODEL, model_family_for
from mlx.modes.image_classification.presentation import (
    display_cam_results,
    display_classification_predictions,
    display_similarity_matches,
    print_config_summary,
    RichImageClassificationReporter,
    resolve_image_dataset_build_request,
)
from mlx.modes.image_classification.requests import (
    BuildImageClassificationDatasetRequest,
    ImageClassificationRequest,
)
from mlx.modes.image_classification.train import (
    SmokeTestImageClassificationModel,
    TrainImageClassificationModel,
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


def _list_models(config: dict[str, Any]):
    summaries = ListImageClassificationModels(config).execute()
    print_model_parameter_table(summaries, title="Image Classification Models")
    return summaries


def _infer_image(config: dict[str, Any]):
    result = InferImageClassification(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute()
    if config.get("display", True):
        if "top_matches" in result:
            display_similarity_matches(result)
        else:
            display_classification_predictions(result)
    return result


def _generate_cams(config: dict[str, Any]):
    results = GenerateImageClassificationCams(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute()
    if config.get("display", True):
        display_cam_results(results, delay=int(config.get("window_delay", 0)))
    return results


ACTION_HANDLERS = {
    "benchmark": lambda config: BenchmarkImageClassification(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute(),
    "build-dataset": lambda config: BuildImageClassificationDataset(
        BuildImageClassificationDatasetRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
        input_resolver=resolve_image_dataset_build_request,
    ).execute(),
    "infer-image": _infer_image,
    "ls-models": _list_models,
    "cam": _generate_cams,
    "test": lambda config: SmokeTestImageClassificationModel(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute(),
    "train": lambda config: TrainImageClassificationModel(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute(),
}


def run_image_classification(mode_config: dict[str, Any]) -> Any:
    if mode_config.get("platform", "local") == "aws":
        from mlx.modes.image_classification.aws.runner import run_aws_image_classification

        return run_aws_image_classification(mode_config)

    config = {**DEFAULT_CONFIG, **mode_config}
    action = config["action"]
    if action == "ls-models":
        return ACTION_HANDLERS[action](config)

    model_name = mode_config.get("model") or DEFAULT_MODEL
    family = model_family_for(model_name)
    if mode_config.get("model") is None and mode_config.get("width") == 256 and mode_config.get("height") == 256:
        if family == "one-shot":
            config["input_size"] = (105, 105)
            config["width"] = 105
            config["height"] = 105
        else:
            config["input_size"] = (224, 224)
            config["width"] = 224
            config["height"] = 224
    else:
        config["input_size"] = tuple(config.get("input_size", (config["width"], config["height"])))
    config["model"] = model_name

    print_config_summary(model_name, family, config)

    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for image-classification. Available actions: {available}."
        )

    return handler(config)
