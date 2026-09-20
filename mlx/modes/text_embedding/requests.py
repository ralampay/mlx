from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlx.core.requests import ConfigRequest


DEFAULT_K_VALUES = (1, 5, 10, 20, 100)


@dataclass(frozen=True)
class EmbedTextRequest(ConfigRequest):
    model: Optional[str] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    vector_store: str = "chroma"
    embedding_backend: str = "llama-cpp"
    query_prefix: str = ""
    document_prefix: str = ""
    normalize_embeddings: bool = False
    batch_size: int = 16
    representation: str = "original"
    adapter: Optional[str] = None
    device: str = "cpu"


@dataclass(frozen=True)
class BenchmarkTextEmbeddingRequest(ConfigRequest):
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    vector_store: Optional[str] = None
    top_k: int = 100
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    metrics: tuple[str, ...] = ("precision", "recall", "mrr", "map", "ndcg")
    representation: Optional[str] = None


__all__ = [
    "BenchmarkTextEmbeddingRequest",
    "DEFAULT_K_VALUES",
    "EmbedTextRequest",
]
