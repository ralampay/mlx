from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_model_parameter_table
from mlx.modes.image_classification.cam import generate_image_classification_cams
from mlx.modes.image_classification.data import build_image_classification_dataset
from mlx.modes.image_classification.evaluation import benchmark_image_classification
from mlx.modes.image_classification.inference import infer_image_classification
from mlx.modes.image_classification.list_models import ListImageClassificationModels
from mlx.modes.image_classification.models import DEFAULT_MODEL, model_family_for
from mlx.modes.image_classification.presentation import print_config_summary
from mlx.modes.image_classification.train import (
    smoke_test_image_classification,
    train_image_classification,
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


ACTION_HANDLERS = {
    "benchmark": benchmark_image_classification,
    "build-dataset": lambda config: build_image_classification_dataset(
        config["dataset_path"],
        train_count=config.get("train_count"),
        val_count=config.get("val_count"),
        test_count=config.get("test_count"),
        train_ratio=config.get("train_ratio"),
        val_ratio=config.get("val_ratio"),
        test_ratio=config.get("test_ratio"),
        split_mode=config.get("split_mode"),
        output_path=config.get("output_path"),
        overwrite=config.get("overwrite", False),
        random_seed=config.get("random_seed"),
    ),
    "infer-image": infer_image_classification,
    "ls-models": _list_models,
    "cam": generate_image_classification_cams,
    "test": smoke_test_image_classification,
    "train": train_image_classification,
}


def run_image_classification(mode_config: dict[str, Any]) -> Any:
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
