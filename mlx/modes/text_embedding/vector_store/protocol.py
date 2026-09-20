from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: Sequence[float]
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Persistent vector index returning best-first, higher-is-better scores."""

    def add(self, records: Sequence[VectorRecord]) -> None:
        ...

    def query(self, vector: Sequence[float], *, k: int) -> Sequence[VectorSearchResult]:
        ...

    def close(self) -> None:
        ...


class VectorStoreFactory(Protocol):
    def __call__(self, path: Path, *, collection: str, create: bool) -> VectorStore:
        ...


__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreFactory",
]
