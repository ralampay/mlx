from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.vector_transforms import VectorRepresentationTransformer
from mlx.modes.text_embedding.artifacts import (
    BenchmarkArtifactWriter,
    EmbeddingArtifactReader,
    EmbeddingArtifactWriter,
    SCHEMA_VERSION,
    prepare_new_output_directory,
    utc_timestamp,
)
from mlx.modes.text_embedding.data import BeirDatasetLoader
from mlx.modes.text_embedding.formatting import RetrievalTextFormatter, resolve_text_formatter
from mlx.modes.text_embedding.embedding.registry import DEFAULT_EMBEDDING_BACKENDS, EmbeddingBackendRegistry
from mlx.modes.text_embedding.metric_registry import DEFAULT_METRIC_REGISTRY, RetrievalMetricRegistry
from mlx.modes.text_embedding.embedding.protocol import TextEmbeddingProvider
from mlx.modes.text_embedding.metrics import (
    QueryMetricResult,
    aggregate_query_metrics,
    compute_query_metrics,
)
from mlx.modes.text_embedding.models import EmbeddedText, document_embedding_text
from mlx.modes.text_embedding.requests import (
    BenchmarkTextEmbeddingRequest,
    EmbedTextRequest,
)
from mlx.modes.text_embedding.vector_store.protocol import (
    VectorRecord,
    VectorStoreFactory,
)
from mlx.modes.text_embedding.vector_store.registry import (
    DEFAULT_VECTOR_STORE_REGISTRY,
    VectorStoreRegistry,
)


@dataclass(frozen=True)
class EmbedTextResult:
    output_dir: Path
    corpus_documents: int
    queries: int
    dimensions: int
    vector_store: str


@dataclass(frozen=True)
class BenchmarkTextEmbeddingResult:
    output_dir: Path
    metrics: Mapping[str, float]
    query_count: int
    failures: int


class EmbedTextCommand:
    def __init__(
        self,
        request: EmbedTextRequest,
        *,
        reporter: WorkflowReporter | None = None,
        dataset_loader: BeirDatasetLoader | None = None,
        provider_factory: Callable[..., TextEmbeddingProvider] | None = None,
        backend_registry: EmbeddingBackendRegistry = DEFAULT_EMBEDDING_BACKENDS,
        vector_store_factory: VectorStoreFactory | None = None,
        vector_store_registry: VectorStoreRegistry = DEFAULT_VECTOR_STORE_REGISTRY,
        artifact_writer: EmbeddingArtifactWriter | None = None,
        transformer: VectorRepresentationTransformer | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.dataset_loader = dataset_loader or BeirDatasetLoader()
        self.backend_registry = backend_registry
        self.provider_factory = provider_factory
        self.vector_store_factory = vector_store_factory
        self.vector_store_registry = vector_store_registry
        self.artifact_writer = artifact_writer or EmbeddingArtifactWriter()
        self.transformer = transformer
        self._dimensions: int | None = None
        self._source_dimensions: int | None = None

    def execute(self) -> EmbedTextResult:
        model_path, input_path = self._validate_request()
        backend = self.backend_registry.resolve(self.request.embedding_backend)
        if self.request.pooling != "auto" and not backend.supports_pooling:
            raise MLXUserError(
                f"Embedding backend '{self.request.embedding_backend}' does not support explicit pooling."
            )
        formatter = resolve_text_formatter(
            self.request.prompt_format,
            query_prefix=self.request.query_prefix,
            document_prefix=self.request.document_prefix,
        )
        provider_factory = self.provider_factory or self.backend_registry.factory(self.request.embedding_backend)
        factory = self.vector_store_factory or self.vector_store_registry.resolve(
            self.request.vector_store
        )
        dataset = self.dataset_loader.load(input_path)
        output_dir = prepare_new_output_directory(
            str(self.request.output_path), purpose="Text embedding"
        )
        started_at = utc_timestamp()
        emit(
            self.reporter,
            "info",
            f"Embedding {len(dataset.corpus)} documents and {len(dataset.queries)} queries.",
            payload={
                "event": "text_embedding_started",
                "documents": len(dataset.corpus),
                "queries": len(dataset.queries),
            },
        )
        options = {"pooling": self.request.pooling} if self.request.pooling != "auto" else {}
        provider = provider_factory(model_path, **options)
        vector_store_path = output_dir / "vector_store"
        vector_store_path.mkdir(parents=True, exist_ok=True)
        store = factory(vector_store_path, collection="corpus", create=True)
        corpus_csv = output_dir / "corpus_embeddings.csv"
        query_csv = output_dir / "query_embeddings.csv"
        try:
            self.artifact_writer.initialize_csv(corpus_csv, kind="corpus")
            self.artifact_writer.initialize_csv(query_csv, kind="query")
            self._embed_corpus(dataset, provider, store, corpus_csv, formatter)
            self._embed_queries(dataset, provider, query_csv, formatter)
        finally:
            store.close()
        dimensions = self._dimensions
        if dimensions is None:
            raise MLXUserError("Embedding provider did not report vector dimensions.")
        if self._source_dimensions is None:
            raise MLXUserError("Embedding provider did not produce source dimensions.")
        self.artifact_writer.write_manifests(
            output_dir,
            dataset=dataset,
            model_path=model_path,
            dimensions=dimensions,
            normalized=self.request.normalize_embeddings,
            vector_store=self.request.vector_store,
            query_prefix=formatter.query_prefix,
            document_prefix=formatter.document_prefix,
            representation=self.request.representation,
            source_dimensions=self._source_dimensions,
            adapter=(dict(self.transformer.provenance) if self.transformer else None),
            started_at=started_at,
            backend=backend.provenance,
            embedding_configuration=self._embedding_configuration(model_path, provider, formatter),
        )
        result = EmbedTextResult(
            output_dir=output_dir,
            corpus_documents=len(dataset.corpus),
            queries=len(dataset.queries),
            dimensions=dimensions,
            vector_store=self.request.vector_store,
        )
        emit(
            self.reporter,
            "success",
            f"Text embedding artifacts written to {output_dir}.",
            payload={"event": "text_embedding_completed", "result": result},
        )
        return result

    def _embedding_configuration(
        self, model_path: Path, provider: TextEmbeddingProvider, formatter: RetrievalTextFormatter
    ) -> dict[str, Any]:
        metadata_reader = getattr(provider, "runtime_metadata", None)
        runtime = metadata_reader() if callable(metadata_reader) else {}
        return {
            "model_path": str(model_path.resolve()),
            "model_filename": model_path.name,
            "pooling_requested": self.request.pooling,
            "pooling_effective": runtime.get("pooling_effective", "unknown"),
            "embedding_dimension": self._source_dimensions,
            "context_length": runtime.get("context_length"),
            "prompt_format_requested": self.request.prompt_format,
            "prompt_format_effective": formatter.effective_format,
            "llama_cpp_python_version": runtime.get("llama_cpp_python_version"),
        }

    def _validate_request(self) -> tuple[Path, Path]:
        if self.request.pooling not in ("auto", "mean", "cls", "last", "none"):
            raise MLXUserError("--pooling must be one of: auto, mean, cls, last, none.")
        if not self.request.model:
            raise MLXUserError("Text embedding requires --model pointing to a model file.")
        if not self.request.input_path:
            raise MLXUserError("Text embedding requires --input pointing to a BEIR dataset.")
        if not self.request.output_path:
            raise MLXUserError("Text embedding requires --output.")
        if self.request.batch_size < 1:
            raise MLXUserError("--batch-size must be at least 1 for text embedding.")
        if not self.request.representation.strip():
            raise MLXUserError("--representation must be non-empty.")
        if self.request.adapter and self.transformer is None:
            raise MLXUserError(
                "An adapter path was supplied without a representation transformer. "
                "Use the text-embedding runner or inject a compatible transformer."
            )
        model = Path(self.request.model).expanduser()
        if not model.is_file():
            raise MLXUserError(f"GGUF embedding model not found: {model}")
        return model, Path(self.request.input_path).expanduser()

    def _embed_corpus(self, dataset, provider, store, path: Path, formatter: RetrievalTextFormatter) -> None:
        total = len(dataset.corpus)
        for start in range(0, total, self.request.batch_size):
            documents = dataset.corpus[start : start + self.request.batch_size]
            texts = [
                formatter.format_document(document_embedding_text(document))
                for document in documents
            ]
            vectors = self._embed_batch(provider, texts)
            records = tuple(
                EmbeddedText(
                    id=document.id,
                    text=document.text,
                    vector=vector,
                    metadata={
                        "document_id": document.id,
                        "title": document.title,
                        "dataset": dataset.name,
                    },
                )
                for document, text, vector in zip(documents, texts, vectors, strict=True)
            )
            self.artifact_writer.append_embeddings(path, records, kind="corpus")
            store.add(
                tuple(
                    VectorRecord(item.id, item.vector, text, item.metadata)
                    for item, text in zip(records, texts, strict=True)
                )
            )
            self._emit_progress("corpus", min(start + len(documents), total), total)

    def _embed_queries(self, dataset, provider, path: Path, formatter: RetrievalTextFormatter) -> None:
        total = len(dataset.queries)
        for start in range(0, total, self.request.batch_size):
            queries = dataset.queries[start : start + self.request.batch_size]
            texts = [formatter.format_query(query.text) for query in queries]
            vectors = self._embed_batch(provider, texts)
            records = tuple(
                EmbeddedText(query.id, query.text, vector, {"dataset": dataset.name})
                for query, vector in zip(queries, vectors, strict=True)
            )
            self.artifact_writer.append_embeddings(path, records, kind="query")
            self._emit_progress("queries", min(start + len(queries), total), total)

    def _embed_batch(self, provider, texts: Sequence[str]) -> list[tuple[float, ...]]:
        source_vectors = provider.embed(texts)
        if len(source_vectors) != len(texts):
            raise MLXUserError(
                f"Embedding provider returned {len(source_vectors)} vectors for {len(texts)} texts."
            )
        source_dimensions = {len(vector) for vector in source_vectors}
        if len(source_dimensions) != 1 or not next(iter(source_dimensions), 0):
            raise MLXUserError("Embedding provider returned inconsistent or empty vectors.")
        current_source_dimensions = next(iter(source_dimensions))
        if provider.dimensions is not None and current_source_dimensions != provider.dimensions:
            raise MLXUserError(
                "Embedding dimensionality mismatch between provider metadata and output vectors."
            )
        if self._source_dimensions is None:
            self._source_dimensions = current_source_dimensions
        elif current_source_dimensions != self._source_dimensions:
            raise MLXUserError(
                f"Source embedding dimensionality mismatch: expected {self._source_dimensions}, "
                f"got {current_source_dimensions}."
            )
        if self.transformer is not None:
            if current_source_dimensions != self.transformer.input_dimensions:
                raise MLXUserError(
                    f"Representation adapter expects {self.transformer.input_dimensions} dimensions, "
                    f"but the embedding provider returned {current_source_dimensions}."
                )
            vectors = self.transformer.transform(source_vectors)
            if len(vectors) != len(source_vectors):
                raise MLXUserError(
                    f"Representation adapter returned {len(vectors)} vectors for "
                    f"{len(source_vectors)} inputs."
                )
        else:
            vectors = source_vectors
        normalized = [self._normalize(vector) for vector in vectors]
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) != 1 or not next(iter(dimensions), 0):
            raise MLXUserError("Representation pipeline returned inconsistent or empty vectors.")
        if self.transformer is not None and dimensions != {self.transformer.output_dimensions}:
            raise MLXUserError(
                "Representation adapter output does not match its declared dimensions."
            )
        current_dimensions = next(iter(dimensions))
        if self._dimensions is None:
            self._dimensions = current_dimensions
        elif current_dimensions != self._dimensions:
            raise MLXUserError(
                f"Embedding dimensionality mismatch: expected {self._dimensions}, got {current_dimensions}."
            )
        return normalized

    def _normalize(self, vector: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in values):
            raise MLXUserError("Embedding provider returned a non-finite vector value.")
        if not self.request.normalize_embeddings:
            return values
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise MLXUserError("Cannot L2-normalize a zero embedding vector.")
        return tuple(value / norm for value in values)

    def _emit_progress(self, phase: str, current: int, total: int) -> None:
        emit(
            self.reporter,
            "progress",
            f"Embedded {current} of {total} {phase}.",
            current=current,
            total=total,
            payload={"event": "text_embedding_progress", "phase": phase},
        )


class BenchmarkTextEmbeddingCommand:
    def __init__(
        self,
        request: BenchmarkTextEmbeddingRequest,
        *,
        reporter: WorkflowReporter | None = None,
        artifact_reader: EmbeddingArtifactReader | None = None,
        metric_registry: RetrievalMetricRegistry = DEFAULT_METRIC_REGISTRY,
        artifact_writer: BenchmarkArtifactWriter | None = None,
        vector_store_factory: VectorStoreFactory | None = None,
        vector_store_registry: VectorStoreRegistry = DEFAULT_VECTOR_STORE_REGISTRY,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.artifact_reader = artifact_reader or EmbeddingArtifactReader()
        self.metric_registry = metric_registry
        self.artifact_writer = artifact_writer or BenchmarkArtifactWriter()
        self.vector_store_factory = vector_store_factory
        self.vector_store_registry = vector_store_registry

    def execute(self) -> BenchmarkTextEmbeddingResult:
        self._validate_request()
        if not self.request.metrics:
            raise MLXUserError("Select at least one retrieval metric.")
        for name in self.request.metrics:
            self.metric_registry.resolve(name)
        artifacts = self.artifact_reader.load(str(self.request.input_path))
        embedding_manifest = artifacts["embedding_manifest"]
        dataset_manifest = artifacts["dataset_manifest"]
        stored_provider = str(embedding_manifest["vector_store"]["provider"])
        requested_provider = self.request.vector_store or stored_provider
        if requested_provider != stored_provider:
            raise MLXUserError(
                "Benchmark vector-store provider does not match the embedding manifest: "
                f"requested {requested_provider}, stored {stored_provider}."
            )
        qrels_by_query = self._qrels_by_query(artifacts["qrels"])
        queries = tuple(item for item in artifacts["queries"] if item[0].id in qrels_by_query)
        if not queries:
            raise MLXUserError("No exported queries have judgments in the selected qrels.")
        excluded_queries = len(artifacts["queries"]) - len(queries)
        corpus_size = int(dataset_manifest["corpus_documents"])
        if corpus_size < 1:
            raise MLXUserError("Embedding artifact manifest has an invalid corpus size.")
        output_dir = prepare_new_output_directory(
            str(self.request.output_path), purpose="Text-embedding benchmark"
        )
        started_at = utc_timestamp()
        factory = self.vector_store_factory or self.vector_store_registry.resolve(requested_provider)
        store = factory(artifacts["root"] / "vector_store", collection="corpus", create=False)
        query_rows = []
        rankings = []
        failures = []
        metric_results: list[QueryMetricResult] = []
        search_depth = min(self.request.top_k, corpus_size)
        try:
            for index, (query, vector) in enumerate(queries, start=1):
                results = tuple(store.query(vector, k=search_depth))
                identifiers = [result.id for result in results]
                if len(results) > search_depth or len(set(identifiers)) != len(identifiers):
                    raise MLXUserError("Vector store returned duplicate IDs or too many results.")
                if any(identifier not in artifacts["corpus_ids"] for identifier in identifiers):
                    raise MLXUserError("Vector store returned an ID absent from the corpus export.")
                if any(not math.isfinite(result.score) for result in results):
                    raise MLXUserError("Vector store returned non-finite scores.")
                if any(results[pos].score < results[pos + 1].score for pos in range(len(results) - 1)):
                    raise MLXUserError("Vector store returned results outside best-first score order.")
                qrels = qrels_by_query.get(query.id, {})
                metrics = compute_query_metrics(
                    [result.id for result in results], qrels, self.request.k_values,
                    metric_names=self.request.metrics, registry=self.metric_registry,
                )
                metric_results.append(metrics)
                query_rows.append(self._query_row(query, metrics))
                rankings.append(self._ranking(query, results, qrels))
                if metrics.best_relevant_rank is None:
                    failures.append(
                        {
                            "query_id": query.id,
                            "query": query.text,
                            "relevant_documents": sorted(
                                identifier for identifier, score in qrels.items() if score > 0
                            ),
                            "best_relevant_rank": None,
                            "reciprocal_rank": 0.0,
                        }
                    )
                emit(
                    self.reporter,
                    "progress",
                    f"Benchmarked query {index} of {len(queries)}.",
                    current=index,
                    total=len(queries),
                    payload={"event": "text_embedding_benchmark_progress"},
                )
        finally:
            store.close()
        aggregate = aggregate_query_metrics(metric_results)
        model = embedding_manifest["model"]
        embedding = embedding_manifest["embedding"]
        representation = self.request.representation or embedding.get("representation", "original")
        embedding_configuration = embedding_manifest.get("embedding_configuration", {})
        summary = {
            "embedding_configuration": embedding_configuration,
            "dataset": dataset_manifest["name"],
            "model": model["path"],
            "model_sha256": model["sha256"],
            "representation": representation,
            "corpus_size": corpus_size,
            "query_count": len(queries),
            "excluded_queries": excluded_queries,
            "dimensions": embedding["dimensions"],
            "similarity": embedding_manifest["vector_store"].get("similarity", "cosine"),
            "vector_store": requested_provider,
            "top_k": self.request.top_k,
            "metrics": aggregate,
        }
        manifest = {
            "embedding_configuration": embedding_configuration,
            "embedding": dict(embedding),
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset_manifest["name"],
            "model_sha256": model["sha256"],
            "embedding_dimensions": embedding["dimensions"],
            "vector_store_provider": requested_provider,
            "similarity": summary["similarity"],
            "k_values": list(self.request.k_values),
            "metrics": list(self.request.metrics),
            "corpus_documents": corpus_size,
            "queries": len(queries),
            "excluded_queries": excluded_queries,
            "relevance_judgments": len(artifacts["qrels"]),
            "benchmark_timestamp": utc_timestamp(),
            "representation": representation,
        }
        self.artifact_writer.write(
            output_dir,
            summary=summary,
            query_rows=query_rows,
            rankings=rankings,
            failures=failures,
            manifest=manifest,
            started_at=started_at,
        )
        emit(
            self.reporter,
            "success",
            f"Text-embedding benchmark written to {output_dir}.",
            payload={"event": "text_embedding_benchmark_completed", "metrics": aggregate},
        )
        return BenchmarkTextEmbeddingResult(
            output_dir=output_dir,
            metrics=aggregate,
            query_count=len(queries),
            failures=len(failures),
        )

    def _validate_request(self) -> None:
        if not self.request.input_path:
            raise MLXUserError("Text-embedding benchmark requires --input.")
        if not self.request.output_path:
            raise MLXUserError("Text-embedding benchmark requires --output.")
        if self.request.top_k < 1:
            raise MLXUserError("--top-k must be at least 1.")
        if not self.request.k_values or any(k < 1 for k in self.request.k_values):
            raise MLXUserError("Benchmark K values must be positive integers.")
        if max(self.request.k_values) > self.request.top_k:
            raise MLXUserError("--top-k must be at least the largest benchmark K value.")

    @staticmethod
    def _qrels_by_query(qrels) -> dict[str, dict[str, int]]:
        values: dict[str, dict[str, int]] = {}
        for item in qrels:
            values.setdefault(item.query_id, {})[item.document_id] = item.relevance
        return values

    @staticmethod
    def _query_row(query, metrics: QueryMetricResult) -> dict[str, Any]:
        row = {
            "query_id": query.id,
            "query": query.text,
            "relevant_count": metrics.relevant_count,
            "retrieved_relevant_at_10": metrics.retrieved_relevant.get(10, 0),
            "recall_at_10": metrics.values.get("recall@10"),
            "reciprocal_rank": metrics.values.get("mrr@10"),
            "ndcg_at_10": metrics.values.get("ndcg@10"),
            "best_relevant_rank": metrics.best_relevant_rank,
        }
        row.update(metrics.values)
        return row

    @staticmethod
    def _ranking(query, results, qrels: Mapping[str, int]) -> dict[str, Any]:
        return {
            "query_id": query.id,
            "query": query.text,
            "results": [
                {
                    "rank": rank,
                    "document_id": result.id,
                    "score": result.score,
                    "relevance": int(qrels.get(result.id, 0)),
                }
                for rank, result in enumerate(results, start=1)
            ],
        }


__all__ = [
    "BenchmarkTextEmbeddingCommand",
    "BenchmarkTextEmbeddingResult",
    "EmbedTextCommand",
    "EmbedTextResult",
]
