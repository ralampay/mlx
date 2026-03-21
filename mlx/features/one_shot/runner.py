from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.features.one_shot.actions import ACTION_HANDLERS, build_model
from mlx.features.one_shot.presentation import print_config_summary

DEFAULT_CONFIG = {
    "action": "test",
    "batch_size": 1,
    "colored": True,
    "dataset_path": "",
    "device": "cpu",
    "embedding_size": 4096,
    "epochs": 100,
    "input_size": (105, 105),
    "num_pairs": 100,
    "refresh_per_second": 2,
}


def run_ic_one_shot(module_config: dict[str, Any]) -> Any:
    config = {**DEFAULT_CONFIG, **{k: v for k, v in module_config.items() if k != "model"}}
    model_name = module_config.get("model", "siamese-le-net")

    print_config_summary(model_name, config)
    net = build_model(model_name, config)

    action = config["action"]
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for ic-one-shot. Available actions: {available}."
        )

    return handler(net, config)
