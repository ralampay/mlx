from __future__ import annotations

from mlx.core.extensions import load_reference, validate_reference
from types import MappingProxyType
from typing import Mapping

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.vector_store.protocol import VectorStoreFactory


class VectorStoreRegistry:
    def __init__(self, factories: Mapping[str, str | VectorStoreFactory] | None = None) -> None:
        self._factories = MappingProxyType({name.strip().lower(): value for name, value in (factories or {}).items()})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve(self, name: str) -> VectorStoreFactory:
        entry = self._factories.get(name.strip().lower())
        if entry is None:
            available = ", ".join(self.names) or "none"
            raise MLXUserError(
                f"Unsupported text-embedding vector store '{name}'. Available providers: {available}."
            )
        if not isinstance(entry, str):
            return entry
        factory = load_reference(entry, kind="vector-store factory")
        if not callable(factory):
            raise MLXUserError(f"Vector-store factory '{entry}' is not callable.")
        return factory

    def register(self, name: str, factory: str | VectorStoreFactory) -> "VectorStoreRegistry":
        name = name.strip().lower()
        if not name:
            raise ValueError("Vector-store name cannot be empty.")
        if isinstance(factory, str):
            validate_reference(factory)
        return VectorStoreRegistry({**self._factories, name: factory})


DEFAULT_VECTOR_STORE_REGISTRY = VectorStoreRegistry(
    {"chroma": "mlx.modes.text_embedding.vector_store.chroma:create_chroma_vector_store"}
)


__all__ = ["DEFAULT_VECTOR_STORE_REGISTRY", "VectorStoreRegistry"]
