from __future__ import annotations

from numbers import Real
from typing import Any, Protocol, Sequence, runtime_checkable

from mlx.core.exceptions import MLXUserError


@runtime_checkable
class TextEmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int | None:
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one fixed-width sequence embedding per input text."""


def validate_sequence_embedding(value: Any, *, context: str) -> list[float]:
    """Normalize a single sequence vector and reject token-level output."""

    if not isinstance(value, (list, tuple)) or not value:
        raise MLXUserError(f"Embedding provider returned no vector for {context}.")
    if any(isinstance(item, (list, tuple)) for item in value):
        raise MLXUserError(
            f"Embedding provider returned token-level embeddings for {context}; "
            "use a GGUF model with sequence pooling."
        )
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise MLXUserError(f"Embedding provider returned a non-numeric vector for {context}.")
    return [float(item) for item in value]


__all__ = ["TextEmbeddingProvider", "validate_sequence_embedding"]
