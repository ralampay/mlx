from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Mapping

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.vector_store.protocol import VectorStoreFactory


class VectorStoreRegistry:
    def __init__(self, factories: Mapping[str, str | VectorStoreFactory] | None = None) -> None:
        self._factories = MappingProxyType(dict(factories or {}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve(self, name: str) -> VectorStoreFactory:
        entry = self._factories.get(name)
        if entry is None:
            available = ", ".join(self.names) or "none"
            raise MLXUserError(
                f"Unsupported text-embedding vector store '{name}'. Available providers: {available}."
            )
        if not isinstance(entry, str):
            return entry
        module_name, attribute = entry.split(":", 1)
        return getattr(import_module(module_name), attribute)

    def register(self, name: str, factory: str | VectorStoreFactory) -> "VectorStoreRegistry":
        return VectorStoreRegistry({**self._factories, name: factory})


DEFAULT_VECTOR_STORE_REGISTRY = VectorStoreRegistry(
    {"chroma": "mlx.modes.text_embedding.vector_store.chroma:create_chroma_vector_store"}
)


__all__ = ["DEFAULT_VECTOR_STORE_REGISTRY", "VectorStoreRegistry"]
