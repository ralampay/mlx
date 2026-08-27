from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any


def build_runtime_config(namespace: argparse.Namespace) -> dict[str, Any]:
    """Normalize parsed CLI values into the legacy runtime configuration shape."""

    config = vars(namespace).copy()
    config.pop("help", None)
    if config.get("mode"):
        config["mode"] = config["mode"].replace("-", "_")
    config["input_size"] = (config["width"], config["height"])
    return config


def explicit_option_destinations(
    parser: argparse.ArgumentParser,
    args: Sequence[str],
) -> set[str]:
    """Return argparse destinations explicitly present on the command line."""

    destinations: set[str] = set()
    option_actions = parser._option_string_actions
    for token in args:
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


__all__ = ["build_runtime_config", "explicit_option_destinations"]
