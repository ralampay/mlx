"""SAR provider contract and immutable lazy factory registry."""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol
from pathlib import Path

from mlx.core.extensions import load_reference
from mlx.core.exceptions import MLXUserError
from .models import Scene
from .requests import SearchRequest


class SarProvider(Protocol):
    def search(self, request: SearchRequest) -> list[Scene]: ...
    def download(self, scene: Scene, directory: Path) -> Path: ...


@dataclass(frozen=True)
class ProviderRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: {"asf": "mlx.modes.sar.asf.provider:create_provider"})

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def create(self, name: str, **kwargs) -> SarProvider:
        if name not in self.entries:
            raise MLXUserError(f"Unknown SAR provider '{name}'. Available: {', '.join(self.entries)}.")
        return load_reference(self.entries[name], kind="SAR provider")(**kwargs)
