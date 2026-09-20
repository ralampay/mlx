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
    assert canonical.actions == ("embed", "benchmark", "ls-metrics", "ls-vector-stores", "ls-embedding-backends")

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


@pytest.mark.parametrize("kind", ["corpus", "query"])
def test_embed_closes_store_when_csv_initialization_fails(tmp_path, kind):
    dataset = write_beir(tmp_path / "dataset")
    model = tmp_path / "model.gguf"
    model.touch()
    store = FakeVectorStore()
    provider = FakeEmbeddingProvider()

    class FailingWriter(EmbeddingArtifactWriter):
        def initialize_csv(self, path, *, kind):
            if kind == failing_kind:
                raise MLXUserError("Cannot initialize embedding CSV")
            return super().initialize_csv(path, kind=kind)

    failing_kind = kind
    with pytest.raises(MLXUserError, match="Cannot initialize"):
        EmbedTextCommand(
            EmbedTextRequest(model=str(model), input_path=str(dataset),
                             output_path=str(tmp_path / "output")),
            provider_factory=lambda path: provider,
            vector_store_factory=lambda *args, **kwargs: store,
            artifact_writer=FailingWriter(),
        ).execute()
    assert store.closed
    assert provider.batches == []


@pytest.mark.parametrize("invalid", ["backend", "store"])
def test_embed_rejects_unknown_components_before_loading_resources(tmp_path, invalid):
    from mlx.modes.text_embedding.embedding.registry import EmbeddingBackendRegistry

    model = tmp_path / "model.gguf"
    model.touch()
    output = tmp_path / "output"
    calls = []
    registry = EmbeddingBackendRegistry({}).register(
        "fake", lambda path: calls.append(path) or FakeEmbeddingProvider(), provenance="fake"
    )
    # Injecting a factory does not bypass the backend's provenance contract.
    with pytest.raises(MLXUserError, match="Unsupported"):
        EmbedTextCommand(
            EmbedTextRequest(
                model=str(model), input_path=str(tmp_path / "unused-dataset"),
                output_path=str(output),
                embedding_backend="missing" if invalid == "backend" else "fake",
                vector_store="missing",
            ),
            backend_registry=registry,
            provider_factory=lambda path: calls.append(path) or FakeEmbeddingProvider(),
        ).execute()
    assert not output.exists()
    assert calls == []


def test_embed_registered_backend_keeps_provenance(tmp_path):
    from mlx.modes.text_embedding.embedding.registry import EmbeddingBackendRegistry

    dataset = write_beir(tmp_path / "dataset")
    model = tmp_path / "model.gguf"
    model.touch()
    output = tmp_path / "output"
    registry = EmbeddingBackendRegistry({}).register(
        "fake", lambda path: FakeEmbeddingProvider(), provenance="custom-embedding-library"
    )
    EmbedTextCommand(
        EmbedTextRequest(model=str(model), input_path=str(dataset), output_path=str(output),
                         embedding_backend="fake"),
        backend_registry=registry,
        vector_store_factory=lambda *args, **kwargs: FakeVectorStore(),
    ).execute()
    manifest = json.loads((output / "embedding_manifest.json").read_text())
    assert manifest["model"]["backend"] == "custom-embedding-library"


@pytest.mark.parametrize("option,values", [
    ("pooling", ("auto", "mean", "cls", "last", "none")),
    ("prompt-format", ("auto", "none", "e5")),
])
def test_embedding_option_choices(option, values):
    from mlx.cli import CLIUsageError

    assert getattr(build_parser().parse_args([]), option.replace("-", "_")) == "auto"
    for value in values:
        parsed = build_parser().parse_args([f"--{option}", value])
        assert getattr(parsed, option.replace("-", "_")) == value
    for invalid in ("invalid", "MEAN", ""):
        with pytest.raises(CLIUsageError, match="invalid choice"):
            build_parser().parse_args([f"--{option}", invalid])


@pytest.fixture
def pooling_binding(monkeypatch):
    from types import SimpleNamespace

    calls = []
    constants = {f"LLAMA_POOLING_TYPE_{name.upper()}": object()
                 for name in ("mean", "cls", "last", "none")}

    class Model:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.resolved = kwargs.get("pooling_type", constants["LLAMA_POOLING_TYPE_CLS"])

        def pooling_type(self):
            return self.resolved

        def n_ctx(self):
            return 512

        def embed(self, texts):
            if self.resolved is constants["LLAMA_POOLING_TYPE_NONE"]:
                return [[[1.0, 2.0], [3.0, 4.0]] for _ in texts]
            return [[1.0, 2.0] for _ in texts]

    binding = SimpleNamespace(Llama=Model, __version__="test-version", **constants)
    monkeypatch.setitem(sys.modules, "llama_cpp", binding)
    return binding, calls


@pytest.mark.parametrize("pooling", ["mean", "cls", "last", "none"])
def test_explicit_pooling_uses_exported_constant(tmp_path, pooling_binding, pooling):
    binding, calls = pooling_binding
    model = tmp_path / "model.gguf"
    model.touch()
    provider = LlamaCppEmbeddingProvider(model, pooling=pooling)
    assert calls == [{"model_path": str(model), "embedding": True,
                      "pooling_type": getattr(binding, f"LLAMA_POOLING_TYPE_{pooling.upper()}")}]
    assert provider.runtime_metadata() == {
        "pooling_effective": pooling, "context_length": 512,
        "llama_cpp_python_version": "test-version",
    }
    if pooling == "none":
        with pytest.raises(MLXUserError, match="token-level.*--pooling mean"):
            provider.embed(["example"])


@pytest.mark.parametrize("kwargs", [{}, {"pooling": "auto"}])
def test_auto_preserves_constructor_and_reports_resolved_pooling(tmp_path, pooling_binding, kwargs):
    _, calls = pooling_binding
    model = tmp_path / "model.gguf"
    model.touch()
    provider = LlamaCppEmbeddingProvider(model, **kwargs)
    assert calls == [{"model_path": str(model), "embedding": True}]
    assert provider.runtime_metadata()["pooling_effective"] == "cls"


@pytest.mark.parametrize("pooling,expected", [("auto", "model/default"), ("mean", "unknown")])
def test_pooling_metadata_does_not_guess(tmp_path, pooling_binding, pooling, expected):
    model = tmp_path / "model.gguf"
    model.touch()
    provider = LlamaCppEmbeddingProvider(model, pooling=pooling, model_factory=lambda **kw: object())
    assert provider.runtime_metadata()["pooling_effective"] == expected
    assert provider.runtime_metadata()["context_length"] is None


def test_missing_pooling_constant_is_actionable(tmp_path, pooling_binding):
    binding, calls = pooling_binding
    del binding.LLAMA_POOLING_TYPE_MEAN
    model = tmp_path / "model.gguf"
    model.touch()
    with pytest.raises(MLXUserError, match="compatible llama-cpp-python"):
        LlamaCppEmbeddingProvider(model, pooling="mean")
    assert not calls


def test_loading_failure_preserves_cause_and_guidance(tmp_path, pooling_binding):
    binding, calls = pooling_binding
    original = RuntimeError("Failed to load model from file")

    def fail(**kwargs):
        calls.append(kwargs)
        raise original

    binding.Llama = fail
    model = tmp_path / "model.gguf"
    model.touch()
    with pytest.raises(MLXUserError) as caught:
        LlamaCppEmbeddingProvider(model)
    assert caught.value.__cause__ is original
    assert len(calls) == 1
    for text in (str(original), "automatic pooling", "--pooling mean", "--pooling cls",
                 "--pooling last", "Do not choose arbitrarily"):
        assert text in str(caught.value)


@pytest.mark.parametrize("requested", ["auto", "none", "e5"])
def test_retrieval_formatter(requested):
    from mlx.modes.text_embedding.formatting import resolve_text_formatter

    formatter = resolve_text_formatter(requested)
    assert formatter.format_query("example") == ("query: example" if requested == "e5" else "example")
    assert formatter.format_document("example") == ("passage: example" if requested == "e5" else "example")


@pytest.mark.parametrize("prefixes", [{"query_prefix": "query: "}, {"document_prefix": "passage: "}])
def test_e5_rejects_custom_prefixes(prefixes):
    from mlx.modes.text_embedding.formatting import resolve_text_formatter

    with pytest.raises(MLXUserError, match="cannot be combined"):
        resolve_text_formatter("e5", **prefixes)
    assert resolve_text_formatter("auto", **prefixes).effective_format == "none"


@pytest.mark.parametrize("legacy", [False, True])
def test_configuration_survives_embedding_and_benchmark(tmp_path, pooling_binding, legacy):
    dataset = write_beir(tmp_path / "scifact")
    model = tmp_path / "e5.gguf"
    model.touch()
    output = tmp_path / "embeddings"
    store = FakeVectorStore()
    EmbedTextCommand(
        EmbedTextRequest(model=str(model), input_path=str(dataset), output_path=str(output),
                         pooling="mean", prompt_format="e5"),
        vector_store_factory=lambda *a, **kw: store,
    ).execute()
    path = output / "embedding_manifest.json"
    manifest = json.loads(path.read_text())
    expected = {
        "model_path": str(model.resolve()), "model_filename": model.name,
        "pooling_requested": "mean", "pooling_effective": "mean",
        "embedding_dimension": 2, "context_length": 512,
        "prompt_format_requested": "e5", "prompt_format_effective": "e5",
        "llama_cpp_python_version": "test-version",
    }
    assert manifest["embedding_configuration"] == expected
    assert json.loads((output / "run_metadata.json").read_text())["embedding_configuration"] == expected
    assert store.records[0].text == "passage: One\nalpha"
    assert manifest["embedding"]["query_prefix"] == "query: "
    if legacy:
        del manifest["embedding_configuration"]
        path.write_text(json.dumps(manifest))
        expected = {}
    results = tmp_path / "results"
    BenchmarkTextEmbeddingCommand(
        BenchmarkTextEmbeddingRequest(input_path=str(output), output_path=str(results), top_k=1, k_values=(1,)),
        vector_store_factory=lambda *a, **kw: FakeVectorStore([VectorSearchResult("d1", 0.9)]),
    ).execute()
    for name in ("benchmark_manifest.json", "metrics.json", "run_metadata.json"):
        assert json.loads((results / name).read_text())["embedding_configuration"] == expected
    with (results / "metrics.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["pooling_effective"] == ("unknown" if legacy else "mean")
    assert row["prompt_format_effective"] == ("unknown" if legacy else "e5")


def test_backend_rejects_unsupported_pooling_before_output(tmp_path):
    from mlx.modes.text_embedding.embedding.registry import EmbeddingBackendRegistry

    model = tmp_path / "model.gguf"
    model.touch()
    output = tmp_path / "output"
    registry = EmbeddingBackendRegistry().register("custom", lambda path: FakeEmbeddingProvider(), provenance="test")
    with pytest.raises(MLXUserError, match="does not support explicit pooling"):
        EmbedTextCommand(
            EmbedTextRequest(model=str(model), input_path="unused", output_path=str(output),
                             embedding_backend="custom", pooling="mean"),
            backend_registry=registry,
        ).execute()
    assert not output.exists()


def test_verbose_cli_preserves_exception_chain(monkeypatch, capsys):
    import mlx.cli as cli

    def fail(config):
        try:
            raise RuntimeError("underlying load failure")
        except RuntimeError as exc:
            raise MLXUserError("model failed") from exc

    monkeypatch.setattr(cli, "resolve_mode_runner", lambda mode: fail)
    assert cli.main(["--mode", "text-embedding", "--action", "embed", "--verbose", "--format", "json"]) == 1
    stderr = capsys.readouterr().err
    assert "Traceback" in stderr
    assert "underlying load failure" in stderr
    assert "model failed" in stderr


@pytest.mark.parametrize("value", [None, [], "mean"])
def test_invalid_configuration_manifest_has_actionable_error(value):
    from mlx.modes.text_embedding.artifacts import EmbeddingArtifactReader

    with pytest.raises(MLXUserError, match="invalid embedding_configuration"):
        EmbeddingArtifactReader._validate_manifests({}, {"embedding_configuration": value})


def test_old_constructor_supports_auto_but_rejects_explicit_pooling(tmp_path, pooling_binding):
    model = tmp_path / "model.gguf"
    model.touch()
    assert LlamaCppEmbeddingProvider(model, model_factory=FakeLlama).embed(["a"]) == [[1.0, 1.0]]
    with pytest.raises(MLXUserError, match="pooling_type support") as caught:
        LlamaCppEmbeddingProvider(model, model_factory=FakeLlama, pooling="mean")
    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize("field,value,message", [
    ("pooling", "invalid", "--pooling must be"),
    ("prompt_format", "invalid", "--prompt-format must be"),
])
def test_python_requests_validate_options(tmp_path, field, value, message):
    model = tmp_path / "model.gguf"
    model.touch()
    output = tmp_path / "output"
    with pytest.raises(MLXUserError, match=message):
        EmbedTextCommand(EmbedTextRequest(
            model=str(model), input_path="unused", output_path=str(output), **{field: value}
        )).execute()
    assert not output.exists()


def test_embedding_failure_preserves_guidance_and_cause(tmp_path, pooling_binding):
    model = tmp_path / "model.gguf"
    model.touch()
    provider = LlamaCppEmbeddingProvider(model)
    original = RuntimeError("sequence embeddings unavailable")

    def fail(texts):
        raise original

    provider._model.embed = fail
    with pytest.raises(MLXUserError, match="--pooling mean") as caught:
        provider.embed(["example"])
    assert caught.value.__cause__ is original


@pytest.mark.parametrize("config", [{"pooling": "mean"}, {"prompt_format": "e5"}])
def test_legacy_csv_rejects_new_nondefault_options(config):
    from mlx.modes.text_embedding.runner import run_text_embedding

    with pytest.raises(MLXUserError, match="legacy CSV"):
        run_text_embedding({"action": "embed", "input_file": "input.csv", **config})
