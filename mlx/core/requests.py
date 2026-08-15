from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfigRequest:
    """Typed request that can losslessly bridge legacy configuration mappings."""

    extras: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        field_names = {item.name for item in fields(cls) if item.name != "extras"}
        values = {name: config[name] for name in field_names if name in config}
        values["extras"] = {name: value for name, value in config.items() if name not in field_names}
        return cls(**values)

    def to_config(self) -> dict[str, Any]:
        values = dict(self.extras)
        for item in fields(self):
            if item.name != "extras":
                values[item.name] = getattr(self, item.name)
        return values

