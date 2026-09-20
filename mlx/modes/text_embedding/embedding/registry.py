from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping
from pathlib import Path
from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference


@dataclass(frozen=True)
class EmbeddingBackend:
    factory: str | Callable
    provenance: str
    supports_pooling: bool = False


@dataclass(frozen=True)
class EmbeddingBackendRegistry:
    entries: Mapping[str, EmbeddingBackend] = field(default_factory=lambda: {
        "llama-cpp": EmbeddingBackend(
            "mlx.modes.text_embedding.embedding.llama_cpp:LlamaCppEmbeddingProvider",
            "llama-cpp-python",
            supports_pooling=True,
        )
    })

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def register(self, name: str, factory, *, provenance: str, supports_pooling: bool = False) -> "EmbeddingBackendRegistry":
        name = name.strip().lower()
        if not name or not provenance:
            raise ValueError("Embedding backends require a name and provenance.")
        return EmbeddingBackendRegistry({**self.entries, name: EmbeddingBackend(factory, provenance, supports_pooling)})

    def resolve(self, name: str) -> EmbeddingBackend:
        try:
            return self.entries[name.strip().lower()]
        except KeyError as exc:
            raise MLXUserError(f"Unsupported embedding backend '{name}'. Available: {', '.join(sorted(self.entries))}.") from exc

    def factory(self, name: str) -> Callable:
        reference = self.resolve(name).factory
        factory = load_reference(reference, kind="embedding backend") if isinstance(reference, str) else reference
        if not callable(factory):
            raise MLXUserError(f"Embedding backend '{name}' must be callable.")
        return factory


DEFAULT_EMBEDDING_BACKENDS = EmbeddingBackendRegistry()
