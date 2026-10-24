"""Declarative CLI metadata shared by mode-owned option inventories."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class CliOption:
    flags: tuple[str, ...]
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(*self.flags, **self.settings)


def boolean_value(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected True or False.")
    return value.lower() == "true"
