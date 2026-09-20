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
        pooling: str = "auto",
    ) -> None:
        if pooling not in ("auto", "mean", "cls", "last", "none"):
            raise MLXUserError("--pooling must be one of: auto, mean, cls, last, none.")
        self.pooling_requested = pooling
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise MLXUserError(f"GGUF embedding model not found: {path}")
        if path.suffix.lower() != ".gguf":
            raise MLXUserError(
                f"Embedding model must be a .gguf file: {path}"
            )
        factory = model_factory or self._import_llama()
        options = {}
        if pooling != "auto":
            try:
                import llama_cpp
                options["pooling_type"] = getattr(llama_cpp, f"LLAMA_POOLING_TYPE_{pooling.upper()}")
            except (ImportError, AttributeError) as exc:
                raise MLXUserError(
                    f"Installed llama-cpp-python does not expose the constant for --pooling {pooling}. "
                    "Install a compatible llama-cpp-python version with pooling_type support."
                ) from exc
        try:
            self._model = factory(model_path=str(path), embedding=True, **options)
        except Exception as exc:
            raise MLXUserError(
                f"Unable to load GGUF embedding model '{path}' with "
                f"{'automatic' if pooling == 'auto' else pooling} pooling: {exc}. "
                "Check model compatibility and llama-cpp-python pooling_type support. "
                + self._pooling_guidance()
            ) from exc
        self._dimensions: int | None = None

    @staticmethod
    def _pooling_guidance() -> str:
        return (
            "The model may require an explicit sequence pooling strategy. "
            "Try --pooling mean, --pooling cls, or --pooling last. "
            "Do not choose arbitrarily for benchmark experiments; use the pooling method "
            "intended by the original model architecture. No token averaging is performed."
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Read actual runtime settings, never infer resolved pooling from the request."""
        effective = "model/default" if self.pooling_requested == "auto" else "unknown"
        version = None
        try:
            import llama_cpp
        except ImportError:
            llama_cpp = None
        if llama_cpp is not None:
            version = getattr(llama_cpp, "__version__", None)
            accessor = getattr(self._model, "pooling_type", None)
            if callable(accessor):
                resolved = accessor()
                for name in ("mean", "cls", "last", "none", "rank"):
                    constant = getattr(llama_cpp, f"LLAMA_POOLING_TYPE_{name.upper()}", None)
                    if constant is not None and resolved == constant:
                        effective = name
                        break
        context_accessor = getattr(self._model, "n_ctx", None)
        return {
            "pooling_effective": effective,
            "context_length": context_accessor() if callable(context_accessor) else None,
            "llama_cpp_python_version": version,
        }

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
                "Check model context limits and GGUF embedding compatibility. "
                + self._pooling_guidance()
            ) from exc
        if not isinstance(raw, (list, tuple)):
            raise MLXUserError("llama.cpp returned an invalid embedding batch.")
        # Some test doubles and older bindings return a bare vector for one item.
        values = [raw] if len(batch) == 1 and raw and not isinstance(raw[0], (list, tuple)) else raw
        if len(values) != len(batch):
            raise MLXUserError(
                f"llama.cpp returned {len(values)} embeddings for {len(batch)} input texts."
            )
        try:
            vectors = [
                validate_sequence_embedding(value, context=f"batch item {index}")
                for index, value in enumerate(values, start=1)
            ]
        except MLXUserError as exc:
            raise MLXUserError(f"{exc} {self._pooling_guidance()}") from exc
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
