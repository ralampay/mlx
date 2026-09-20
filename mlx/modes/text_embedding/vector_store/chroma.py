from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.vector_store.protocol import (
    VectorRecord,
    VectorSearchResult,
)


class ChromaVectorStore:
    """Persistent cosine index that accepts caller-generated embeddings."""

    def __init__(
        self,
        path: str | Path,
        *,
        collection: str = "corpus",
        create: bool = True,
        client_factory=None,
    ) -> None:
        root = Path(path).expanduser()
        if not create and not root.is_dir():
            raise MLXUserError(f"Chroma vector-store directory not found: {root}")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        factory = client_factory or self._import_client_factory()
        try:
            self._client = factory(path=str(root))
            if create:
                self._collection = self._client.get_or_create_collection(
                    name=collection,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )
            else:
                self._collection = self._client.get_collection(
                    name=collection,
                    embedding_function=None,
                )
                metadata = self._collection.metadata or {}
                if metadata.get("hnsw:space") != "cosine":
                    raise MLXUserError("Stored Chroma collection does not declare cosine similarity.")
        except MLXUserError:
            raise
        except Exception as exc:
            raise MLXUserError(
                f"Unable to open Chroma vector store '{root}': {exc}."
            ) from exc

    @staticmethod
    def _import_client_factory():
        try:
            import chromadb
        except ImportError as exc:
            raise MLXUserError(
                "Chroma vector storage requires chromadb. Install MLX with the "
                "'text-embedding' extra."
            ) from exc
        return chromadb.PersistentClient

    def add(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        try:
            self._collection.add(
                ids=[record.id for record in records],
                embeddings=[list(record.vector) for record in records],
                documents=[record.text for record in records],
                metadatas=[dict(record.metadata) for record in records],
            )
        except Exception as exc:
            raise MLXUserError(f"Unable to add vectors to Chroma: {exc}") from exc

    def query(self, vector: Sequence[float], *, k: int) -> Sequence[VectorSearchResult]:
        if k < 1:
            raise MLXUserError("Vector search k must be at least 1.")
        try:
            raw: dict[str, Any] = self._collection.query(
                query_embeddings=[list(vector)],
                n_results=k,
                include=["distances", "metadatas"],
            )
        except Exception as exc:
            raise MLXUserError(f"Unable to query Chroma vector store: {exc}") from exc
        ids = (raw.get("ids") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        metadata = (raw.get("metadatas") or [[]])[0]
        return tuple(
            VectorSearchResult(
                id=str(identifier),
                score=1.0 - float(distance),
                metadata=dict(item or {}),
            )
            for identifier, distance, item in zip(ids, distances, metadata, strict=True)
        )

    def close(self) -> None:
        # PersistentClient writes synchronously and currently has no public close API.
        return None


def create_chroma_vector_store(
    path: Path, *, collection: str = "corpus", create: bool = True
) -> ChromaVectorStore:
    return ChromaVectorStore(path, collection=collection, create=create)


__all__ = ["ChromaVectorStore", "create_chroma_vector_store"]
