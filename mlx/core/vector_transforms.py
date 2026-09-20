from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class VectorRepresentationTransformer(Protocol):
    """Batch transform for portable numeric vector representations."""

    @property
    def input_dimensions(self) -> int:
        ...

    @property
    def output_dimensions(self) -> int:
        ...

    @property
    def provenance(self) -> Mapping[str, Any]:
        ...

    def transform(self, vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        ...


__all__ = ["VectorRepresentationTransformer"]
