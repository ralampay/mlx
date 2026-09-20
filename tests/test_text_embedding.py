from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import pytest

from mlx.cli import _build_config, build_parser
from mlx.cli_routing import resolve_mode_descriptor
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.artifacts import EmbeddingArtifactWriter
from mlx.modes.text_embedding.commands import (
    BenchmarkTextEmbeddingCommand,
    EmbedTextCommand,
)
from mlx.modes.text_embedding.data import BeirDatasetLoader
from mlx.modes.text_embedding.embedding.llama_cpp import LlamaCppEmbeddingProvider
from mlx.modes.text_embedding.metrics import (
    average_precision_at_k,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from mlx.modes.text_embedding.requests import (
    BenchmarkTextEmbeddingRequest,
    EmbedTextRequest,
)
from mlx.modes.text_embedding.vector_store.chroma import ChromaVectorStore
from mlx.modes.text_embedding.vector_store.protocol import (
    VectorRecord,
    VectorSearchResult,
    VectorStore,
)


def write_beir(root: Path) -> Path:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        '\n'.join([
            json.dumps({"_id": "d1", "title": "One", "text": "alpha"}),
            json.dumps({"_id": "d2", "title": "", "text": "beta"}),
            json.dumps({"_id": "d3", "title": "Three", "text": "gamma"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '\n'.join([
            json.dumps({"_id": "q1", "text": "alpha query"}),
            json.dumps({"_id": "q2", "text": "missing query"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\n",
        encoding="utf-8",
    )
    return root


def test_mode_aliases_and_exact_cli_forms_parse() -> None:
    canonical = resolve_mode_descriptor("text_embedding")
    assert resolve_mode_descriptor("text-embedding") is canonical
    assert resolve_mode_descriptor("nlp") is canonical
    assert canonical.actions == ("embed", "benchmark")

    embed = _build_config(build_parser().parse_args(
        "--mode text-embedding --action embed --model model.gguf --input ./dataset --output ./output".split()
    ))
    benchmark = _build_config(build_parser().parse_args(
        "--mode text-embedding --action benchmark --input ./output --output ./benchmark".split()
    ))
    assert embed["mode"] == "text_embedding"
    assert embed["input_path"] == "./dataset"
    assert benchmark["action"] == "benchmark"


def test_beir_dataset_loader_validates_and_normalizes(tmp_path: Path) -> None:
    dataset = BeirDatasetLoader().load(write_beir(tmp_path / "scifact"))
    assert dataset.name == "scifact"
    assert dataset.corpus[0].title == "One"
    assert dataset.queries[0].id == "q1"
    assert dataset.qrels[0].document_id == "d1"


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("corpus.jsonl", "corpus.jsonl"),
        ("queries.jsonl", "queries.jsonl"),
        ("qrels/test.tsv", "qrels/test.tsv"),
    ],
)
def test_beir_dataset_loader_rejects_missing_files(tmp_path, relative, message) -> None:
    root = write_beir(tmp_path / "dataset")
    (root / relative).unlink()
    with pytest.raises(MLXUserError, match=message):
        BeirDatasetLoader().load(root)


def test_beir_dataset_loader_rejects_bad_json_duplicates_and_unknown_qrels(tmp_path) -> None:
    root = write_beir(tmp_path / "bad-json")
    (root / "corpus.jsonl").write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(MLXUserError, match="Malformed JSONL"):
        BeirDatasetLoader().load(root)

    root = write_beir(tmp_path / "duplicates")
    duplicate = json.dumps({"_id": "d1", "title": "", "text": "again"})
    with (root / "corpus.jsonl").open("a", encoding="utf-8") as output:
        output.write(duplicate + "\n")
    with pytest.raises(MLXUserError, match="Duplicate corpus"):
        BeirDatasetLoader().load(root)

    root = write_beir(tmp_path / "duplicate-query")
    with (root / "queries.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps({"_id": "q1", "text": "again"}) + "\n")
    with pytest.raises(MLXUserError, match="Duplicate query"):
        BeirDatasetLoader().load(root)

    root = write_beir(tmp_path / "unknown-doc")
    (root / "qrels" / "test.tsv").write_text("q1\tmissing\t1\n", encoding="utf-8")
    with pytest.raises(MLXUserError, match="unknown corpus"):
        BeirDatasetLoader().load(root)

    root = write_beir(tmp_path / "unknown-query")
    (root / "qrels" / "test.tsv").write_text("missing\td1\t1\n", encoding="utf-8")
    with pytest.raises(MLXUserError, match="unknown query"):
        BeirDatasetLoader().load(root)


class FakeLlama:
    def __init__(self, *, model_path, embedding):
        self.model_path = model_path
        self.embedding = embedding

    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def test_llama_cpp_provider_batches_and_validates_dimensions(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    provider = LlamaCppEmbeddingProvider(model, model_factory=FakeLlama)
    assert provider.embed(["a", "bbb"]) == [[1.0, 1.0], [3.0, 1.0]]
    assert provider.dimensions == 2

    provider._model.embed = lambda _texts: [[1.0], [2.0, 3.0]]
    with pytest.raises(MLXUserError, match="dimensionality mismatch"):
        provider.embed(["a", "b"])


def test_llama_cpp_provider_rejects_token_vectors_and_load_errors(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    provider = LlamaCppEmbeddingProvider(model, model_factory=FakeLlama)
    provider._model.embed = lambda _texts: [[[1.0], [2.0]]]
    with pytest.raises(MLXUserError, match="token-level"):
        provider.embed(["a"])

    def broken(**_kwargs):
        raise RuntimeError("bad model")

    with pytest.raises(MLXUserError, match="Unable to load GGUF"):
        LlamaCppEmbeddingProvider(model, model_factory=broken)


def test_llama_cpp_provider_reports_missing_dependency(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(MLXUserError, match="text-embedding.*extra"):
        LlamaCppEmbeddingProvider(model)


class FakeEmbeddingProvider:
    dimensions = 2

    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(tuple(texts))
        return [[3.0, 4.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, rankings=None):
        self.records = []
        self.rankings = list(rankings or [])
        self.closed = False

    def add(self, records):
        self.records.extend(records)

    def query(self, vector, *, k):
        return tuple(self.rankings[:k])

    def close(self):
        self.closed = True


def test_fake_vector_store_satisfies_runtime_contract() -> None:
    assert isinstance(FakeVectorStore(), VectorStore)


def test_embed_command_writes_artifacts_prefixes_normalizes_and_indexes(tmp_path: Path) -> None:
    dataset = write_beir(tmp_path / "scifact")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    output = tmp_path / "artifacts"
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()

    result = EmbedTextCommand(
        EmbedTextRequest(
            model=str(model), input_path=str(dataset), output_path=str(output),
            query_prefix="query: ", document_prefix="passage: ",
            normalize_embeddings=True, batch_size=2,
        ),
        provider_factory=lambda _path: provider,
        vector_store_factory=lambda *_args, **_kwargs: store,
    ).execute()

    assert result.dimensions == 2
    assert len(store.records) == 3
    assert store.records[0].id == "d1"
    assert store.records[0].text == "passage: One\nalpha"
    assert store.records[0].vector == pytest.approx((0.6, 0.8))
    assert provider.batches[-1][0] == "query: alpha query"
    assert store.closed
    assert {path.name for path in output.iterdir()} >= {
        "corpus_embeddings.csv", "query_embeddings.csv", "vector_store",
        "dataset_manifest.json", "embedding_manifest.json", "run_metadata.json",
    }
    manifest = json.loads((output / "embedding_manifest.json").read_text())
    assert manifest["embedding"]["query_prefix"] == "query: "
    assert manifest["embedding"]["normalized"] is True
    with (output / "corpus_embeddings.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["id"] for row in rows] == ["d1", "d2", "d3"]
    assert rows[0]["title"] == "One"
    assert rows[0]["text"] == "alpha"
    assert json.loads(rows[0]["embedding"]) == pytest.approx([0.6, 0.8])


def test_retrieval_metric_formulas_are_hand_verifiable() -> None:
    relevances = [1, 0, 1]
    assert precision_at_k(relevances, 2) == 0.5
    assert recall_at_k(relevances, 2, 2) == 0.5
    assert reciprocal_rank_at_k([0, 1], 2) == 0.5
    assert average_precision_at_k(relevances, 2, 3) == pytest.approx((1 + 2 / 3) / 2)
    assert dcg_at_k([3, 2], 2) == pytest.approx(7 + 3 / math.log2(3))
    assert ndcg_at_k([2, 3], [3, 2], 2) < 1.0
    assert ndcg_at_k([3, 2], [3, 2], 2) == 1.0


def test_benchmark_command_writes_metrics_rankings_and_failures(tmp_path: Path) -> None:
    dataset = write_beir(tmp_path / "scifact")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    artifacts = tmp_path / "artifacts"
    provider = FakeEmbeddingProvider()
    embed_store = FakeVectorStore()
    EmbedTextCommand(
        EmbedTextRequest(model=str(model), input_path=str(dataset), output_path=str(artifacts)),
        provider_factory=lambda _path: provider,
        vector_store_factory=lambda *_args, **_kwargs: embed_store,
    ).execute()

    benchmark_store = FakeVectorStore([
        VectorSearchResult("d1", 0.9),
        VectorSearchResult("d2", 0.5),
    ])
    output = tmp_path / "results"
    result = BenchmarkTextEmbeddingCommand(
        BenchmarkTextEmbeddingRequest(
            input_path=str(artifacts), output_path=str(output), top_k=2, k_values=(1, 2)
        ),
        vector_store_factory=lambda *_args, **_kwargs: benchmark_store,
    ).execute()

    assert result.metrics["recall@1"] == 0.5
    assert result.metrics["mrr@2"] == 0.5
    assert result.failures == 1
    assert {path.name for path in output.iterdir()} == {
        "metrics.json", "metrics.csv", "query_metrics.csv", "rankings.jsonl",
        "failures.csv", "benchmark_manifest.json", "run_metadata.json", "report.md",
    }
    rankings = [json.loads(line) for line in (output / "rankings.jsonl").read_text().splitlines()]
    assert rankings[0]["results"][0]["document_id"] == "d1"
    with (output / "query_metrics.csv").open(newline="") as source:
        query_metrics = list(csv.DictReader(source))
    assert float(query_metrics[0]["recall@1"]) == 1.0
    assert float(query_metrics[1]["recall@1"]) == 0.0
    with (output / "failures.csv").open(newline="") as source:
        assert [row["query_id"] for row in csv.DictReader(source)] == ["q2"]


class FakeCollection:
    def __init__(self):
        self.added = None

    def add(self, **kwargs):
        self.added = kwargs

    def query(self, **_kwargs):
        return {"ids": [["d1"]], "distances": [[0.25]], "metadatas": [[{"title": "One"}]]}


class FakeChromaClient:
    collection = FakeCollection()

    def __init__(self, *, path):
        self.path = path

    def get_or_create_collection(self, **_kwargs):
        return self.collection

    def get_collection(self, **_kwargs):
        return self.collection


def test_chroma_adapter_passes_vectors_metadata_and_normalizes_scores(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "vectors", client_factory=FakeChromaClient)
    store.add([VectorRecord("d1", [1.0, 0.0], "text", {"title": "One"})])
    assert FakeChromaClient.collection.added["embeddings"] == [[1.0, 0.0]]
    result = store.query([1.0, 0.0], k=1)[0]
    assert result == VectorSearchResult("d1", 0.75, {"title": "One"})


def test_chroma_adapter_reports_missing_dependency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(MLXUserError, match="requires chromadb"):
        ChromaVectorStore(tmp_path / "vectors")


def test_chroma_adapter_persists_and_reloads_when_installed(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    path = tmp_path / "persistent"
    first = ChromaVectorStore(path)
    first.add([
        VectorRecord("d1", [1.0, 0.0], "one", {"document_id": "d1", "title": "One"}),
        VectorRecord("d2", [0.0, 1.0], "two", {"document_id": "d2", "title": "Two"}),
    ])
    first.close()

    second = ChromaVectorStore(path, create=False)
    results = second.query([1.0, 0.0], k=2)
    second.close()
    assert [item.id for item in results] == ["d1", "d2"]
    assert results[0].metadata["title"] == "One"
    assert results[0].score > results[1].score
