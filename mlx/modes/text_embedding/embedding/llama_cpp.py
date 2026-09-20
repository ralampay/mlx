from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.embedding.protocol import validate_sequence_embedding


class LlamaCppEmbeddingProvider:
    """Lazy llama-cpp-python adapter exposing only sequence-level embeddings."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise MLXUserError(f"GGUF embedding model not found: {path}")
        if path.suffix.lower() != ".gguf":
            raise MLXUserError(
                f"Embedding model must be a .gguf file: {path}"
            )
        factory = model_factory or self._import_llama()
        try:
            self._model = factory(model_path=str(path), embedding=True)
        except Exception as exc:
            raise MLXUserError(
                f"Unable to load GGUF embedding model '{path}': {exc}. "
                "Check that it supports sequence embeddings."
            ) from exc
        self._dimensions: int | None = None

    @staticmethod
    def _import_llama():
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise MLXUserError(
                "Text embedding requires llama-cpp-python. Install MLX with the "
                "'text-embedding' extra (or the compatible 'nlp' extra)."
            ) from exc
        return Llama

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        if any(not isinstance(text, str) or not text for text in batch):
            raise MLXUserError("Embedding inputs must be non-empty strings.")
        try:
            raw = self._model.embed(batch)
        except Exception as exc:
            raise MLXUserError(
                f"llama.cpp embedding failed for a batch of {len(batch)} text(s): {exc}. "
                "Check model context limits and GGUF embedding compatibility."
            ) from exc
        if not isinstance(raw, (list, tuple)):
            raise MLXUserError("llama.cpp returned an invalid embedding batch.")
        # Some test doubles and older bindings return a bare vector for one item.
        values = [raw] if len(batch) == 1 and raw and not isinstance(raw[0], (list, tuple)) else raw
        if len(values) != len(batch):
            raise MLXUserError(
                f"llama.cpp returned {len(values)} embeddings for {len(batch)} input texts."
            )
        vectors = [
            validate_sequence_embedding(value, context=f"batch item {index}")
            for index, value in enumerate(values, start=1)
        ]
        for vector in vectors:
            if self._dimensions is None:
                self._dimensions = len(vector)
            elif len(vector) != self._dimensions:
                raise MLXUserError(
                    "Embedding dimensionality mismatch: "
                    f"expected {self._dimensions}, got {len(vector)}."
                )
        return vectors


__all__ = ["LlamaCppEmbeddingProvider"]
