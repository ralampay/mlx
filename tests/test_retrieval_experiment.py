from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.data import BeirDatasetLoader
from mlx.modes.text_embedding.embedding.llama_cpp import LlamaCppEmbeddingProvider
from mlx.modes.text_embedding.formatting import resolve_text_formatter
from mlx.modes.text_embedding.experiment_artifacts import ExperimentStages
from mlx.modes.text_embedding.experiment_statistics import paired_mean_test, holm_adjust
from mlx.modes.text_embedding.retrieval_datasets import PrepareRetrievalDatasets, RetrievalDatasetSource
from mlx.modes.text_embedding.vector_store.exact import ExactCosineVectorStore
from mlx.modes.text_embedding.vector_store.protocol import VectorRecord


def test_exact_cosine_persistence_ties_and_validation(tmp_path):
    store = ExactCosineVectorStore(tmp_path / "index")
    store.add([VectorRecord("b", [1, 0], ""), VectorRecord("a", [2, 0], ""),
               VectorRecord("c", [-1, 0], ""), VectorRecord("zero", [0, 0], "")])
    with pytest.raises(MLXUserError, match="unique"):
        store.add([VectorRecord("a", [0, 1], "")])
    with pytest.raises(MLXUserError, match="dimension"):
        store.add([VectorRecord("bad", [0, 1, 2], "")])
    store.close()
    loaded = ExactCosineVectorStore(tmp_path / "index", create=False)
    matches = loaded.query([5, 0], k=10)
    assert [m.id for m in matches] == ["a", "b", "zero", "c"]
    assert [m.score for m in matches] == pytest.approx([1, 1, 0, -1])
    with pytest.raises(MLXUserError, match="non-finite"):
        loaded.query([float("nan"), 0], k=1)
    loaded.close()


def test_gemma_format_preserves_titles_and_legacy_composition():
    formatter = resolve_text_formatter("embeddinggemma")
    assert formatter.format_query("question") == "task: search result | query: question"
    assert formatter.format_document("body", title="Title") == "title: Title | text: body"
    assert formatter.format_document("", title="") == "title: none | text: "
    assert resolve_text_formatter("e5").format_document("body", title="Title") == "passage: Title\nbody"
    with pytest.raises(MLXUserError, match="prefix"):
        resolve_text_formatter("embeddinggemma", query_prefix="custom")


def test_qwen3_retrieval_instruction_applies_to_queries_only():
    formatter = resolve_text_formatter("qwen3")
    assert formatter.format_query("question").startswith("Instruct: Given a web search query")
    assert formatter.format_query("question").endswith("Query: question")
    assert formatter.format_document("body", title="Title") == "Title\nbody"


def test_context_length_reaches_binding_and_provider_closes(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    options = {}
    class Binding:
        def __init__(self, **kwargs):
            options.update(kwargs)
        def close(self):
            options["closed"] = True
    provider = LlamaCppEmbeddingProvider(model, model_factory=Binding, context_length=2048)
    assert options["n_ctx"] == options["n_batch"] == options["n_ubatch"] == 2048
    provider.close()
    assert options["closed"]


class FixtureSource:
    def fetch(self, source, destination):
        destination.mkdir()
    def rows(self, directory, kind):
        if kind == "corpus":
            return [{"_id": "d", "text": ""}, {"_id": "e", "text": "nonempty"}]
        if kind == "queries":
            return [{"_id": "q", "text": "question"}]
        return [{"query-id": "q", "corpus-id": "d"}]


def test_conversion_retains_empty_text_qrels_and_reuses_verified_data(tmp_path):
    spec = RetrievalDatasetSource("fixture", "fixture/repo", "revision", 2, 1)
    command = PrepareRetrievalDatasets(tmp_path, source=FixtureSource(), sources=(spec,))
    command.execute()
    command.execute()
    dataset = BeirDatasetLoader(allow_empty_documents=True).load(tmp_path / "fixture")
    assert dataset.corpus[0].text == ""
    assert dataset.qrels[0].relevance == 1
    with pytest.raises(MLXUserError, match="non-empty"):
        BeirDatasetLoader().load(tmp_path / "fixture")
    (tmp_path / "fixture/queries.jsonl").write_text("changed")
    with pytest.raises(MLXUserError, match="conflicts"):
        command.execute()


def test_held_out_suite_selects_pinned_sources(tmp_path):
    command = PrepareRetrievalDatasets(tmp_path, suite="held-out-nano-v1", source=FixtureSource())
    assert [source.name for source in command.sources] == [
        "nano-climatefever", "nano-fever", "nano-nq",
    ]
    assert all(len(source.revision) == 40 for source in command.sources)


def test_preparation_failure_does_not_publish(tmp_path):
    class DuplicateSource(FixtureSource):
        def rows(self, directory, kind):
            values = super().rows(directory, kind)
            return values * 2 if kind == "qrels" else values
    spec = RetrievalDatasetSource("fixture", "fixture/repo", "revision", 2, 1)
    with pytest.raises(MLXUserError, match="Duplicate"):
        PrepareRetrievalDatasets(tmp_path, source=DuplicateSource(), sources=(spec,)).execute()
    assert not list(tmp_path.iterdir())


def test_noninferiority_direction_holm_and_degenerate_case():
    retained = paired_mean_test(np.linspace(-0.002, 0.002, 10), null_mean=-0.01, alpha=0.05)
    degraded = paired_mean_test(np.linspace(-0.032, -0.028, 10), null_mean=-0.01, alpha=0.05)
    assert retained["p_value"] < 0.001
    assert retained["lower_one_sided"] > -0.01
    assert degraded["p_value"] > 0.99
    assert degraded["ci_high"] < -0.01
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert paired_mean_test([0, 0], null_mean=-0.01, alpha=0.05)["degenerate_variance"]
    with pytest.raises(MLXUserError):
        paired_mean_test([0, float("nan")], null_mean=-0.01, alpha=0.05)


def test_stage_resume_retries_failures_and_rejects_modified_inputs(tmp_path):
    identity = {"config": {"seeds": (1, 2)}}
    stages = ExperimentStages(tmp_path / "run", identity, resume=True)
    attempts = []
    def operation(path):
        path.mkdir()
        attempts.append(1)
        (path / "value").write_text("value")
    completed = stages.run("stage", operation)
    ExperimentStages(tmp_path / "run", identity, resume=True).run("stage", operation)
    assert len(attempts) == 1
    with pytest.raises(MLXUserError, match="changed"):
        ExperimentStages(tmp_path / "run", {"changed": True}, resume=True)
    (completed / "value").write_text("tampered")
    with pytest.raises(MLXUserError, match="changed"):
        stages.run("stage", operation)
    def fail(path):
        raise ValueError("interrupted")
    with pytest.raises(MLXUserError, match="interrupted"):
        stages.run("retry", fail)
    stages.run("retry", operation)
    assert len(attempts) == 2


def write_dataset(root):
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text("\n".join(json.dumps({"_id": f"d{i}", "title": "Title", "text": f"body {i}"}) for i in range(10)) + "\n")
    (root / "queries.jsonl").write_text(json.dumps({"_id": "d0", "text": "query zero"}) + "\n" + json.dumps({"_id": "q1", "text": "query one"}) + "\n")
    (root / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nd0\td1\t1\nq1\td2\t1\n")


def test_end_to_end_ablation_uses_cached_vectors_and_corpus_only(tmp_path):
    from mlx.modes.autoencoder.commands import TrainAutoencoder
    from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
    from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
    from mlx.modes.text_embedding.commands import EmbedTextCommand
    from mlx.modes.text_embedding.experiment import BenchmarkAutoencoderRetrieval
    from mlx.modes.text_embedding.experiment_requests import AutoencoderRetrievalRequest
    for name in ("one", "two"):
        write_dataset(tmp_path / "datasets" / name)
    model = tmp_path / "model.gguf"
    model.touch()
    calls, training_inputs = [], []
    class Provider:
        dimensions = 4
        def embed(self, texts):
            calls.extend(texts)
            return [[len(t), sum(map(ord, t)) % 11, 1, sum(map(ord, t)) % 7] for t in texts]
    def embed(request, **kwargs):
        return EmbedTextCommand(request, provider_factory=lambda *a, **k: Provider(), **kwargs)
    def train(options):
        training_inputs.append(Path(options["input_path"]).name)
        return TrainAutoencoder(AutoencoderTrainRequest(**options))
    request = AutoencoderRetrievalRequest(dataset_path=str(tmp_path / "datasets"), output_path=str(tmp_path / "run"),
                                        model=str(model), bottleneck_dims=(2,), hidden_dim=3,
                                        seeds=(1, 2), epochs=1, batch_size=3, resume=True)
    def command(r=request):
        return BenchmarkAutoencoderRetrieval(r, training_factory=train,
                    transformer_factory=AutoencoderRepresentationTransformer, embed_factory=embed, datasets=("one", "two"))
    command().execute()
    assert len(calls) == 24  # No GGUF calls for any compressed variant.
    assert training_inputs == ["corpus_embeddings.csv"] * 8
    summary = json.loads((tmp_path / "run/comparison/statistics.json").read_text())
    assert len(summary["primary"]) == 2
    assert all(row["datasets"] == 2 for row in summary["primary"])
    rankings = [json.loads(line) for line in (tmp_path / "run/one/baseline/rankings.jsonl").read_text().splitlines()]
    assert all(row["document_id"] != "d0" for row in rankings[0]["results"])
    command().execute()
    assert len(calls) == 24 and len(training_inputs) == 8
    with pytest.raises(MLXUserError, match="changed"):
        command(replace(request, similarity_weight=2.0)).execute()


def test_real_gguf_retrieval_smoke(tmp_path):
    import os
    model = os.environ.get("MLX_RETRIEVAL_SMOKE_MODEL")
    if not model:
        pytest.skip("Set MLX_RETRIEVAL_SMOKE_MODEL to run the optional real-GGUF smoke test.")
    from mlx.modes.autoencoder.commands import TrainAutoencoder
    from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
    from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
    from mlx.modes.text_embedding.experiment import BenchmarkAutoencoderRetrieval
    from mlx.modes.text_embedding.experiment_requests import AutoencoderRetrievalRequest
    for name in ("one", "two"):
        write_dataset(tmp_path / "datasets" / name)
    request = AutoencoderRetrievalRequest(dataset_path=str(tmp_path / "datasets"), output_path=str(tmp_path / "run"),
                                         model=model, bottleneck_dims=(128,), seeds=(42, 43), epochs=1)
    result = BenchmarkAutoencoderRetrieval(
        request, datasets=("one", "two"),
        training_factory=lambda options: TrainAutoencoder(AutoencoderTrainRequest(**options)),
        transformer_factory=AutoencoderRepresentationTransformer).execute()
    assert Path(result["report"]).is_file()
    manifest = json.loads((tmp_path / "run/one/original/embedding_manifest.json").read_text())
    assert manifest["embedding"]["dimensions"] == 768
    assert manifest["embedding_configuration"]["context_length"] == 2048


def test_loss_ablations_match_initialization_and_batch_order(tmp_path):
    import csv
    import torch
    from mlx.modes.autoencoder.commands import TrainAutoencoder
    from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
    path = tmp_path / "vectors.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "embedding"])
        writer.writerows((str(i), json.dumps([i, i + 1, i + 2, i + 3])) for i in range(11))
    class TraceTraining(TrainAutoencoder):
        def _train_epoch(self, model, loader, criterion, optimizer):
            self.initial = {key: value.clone() for key, value in model.state_dict().items()}
            self.batches = [batch.tolist() for batch in loader]
            return 0.0
        def _validation_loss(self, model, loader, criterion):
            self.validation = [batch.tolist() for batch in loader]
            return 0.0
    traces = []
    for loss in ("mse", "mse-similarity"):
        command = TraceTraining(AutoencoderTrainRequest(
            input_path=str(path), output_path=str(tmp_path / loss), hidden_dim=3, bottleneck_dim=2,
            loss=loss, minimum_batch_size=2, epochs=1, batch_size=4, plots=False))
        command.execute()
        traces.append(command)
    assert traces[0].batches == traces[1].batches
    assert traces[0].validation == traces[1].validation
    assert sorted(map(len, traces[0].batches)) == [4, 5]
    assert all(torch.equal(value, traces[1].initial[key]) for key, value in traces[0].initial.items())


def test_experiment_cli_preserves_explicit_flags_and_uses_action_defaults(monkeypatch, tmp_path):
    from mlx.cli import build_parser, _build_config
    from mlx.cli_config import explicit_option_destinations
    from mlx.modes.text_embedding.runner import run_text_embedding
    from mlx.modes.text_embedding.experiment import BenchmarkAutoencoderRetrieval
    parser = build_parser()
    arguments = ["--mode", "text-embedding", "--action", "benchmark-autoencoders", "--model", "model.gguf",
                 "--dataset-path", str(tmp_path), "--output", str(tmp_path / "output"),
                 "--bottleneck-dims", "32,64", "--seeds", "7,8", "--resume", "--format", "json"]
    config = _build_config(parser.parse_args(arguments))
    config["_explicit_options"] = explicit_option_destinations(parser, arguments)
    monkeypatch.setattr(BenchmarkAutoencoderRetrieval, "execute", lambda self: self.request)
    request = run_text_embedding(config)
    assert request.epochs == 50 and request.batch_size == 64
    assert request.lr == 0.001 and request.val_ratio == 0.2
    assert request.normalize_embeddings and request.exclude_self_matches and request.resume
    assert request.bottleneck_dims == (32, 64) and request.seeds == (7, 8)
    assert request.vector_store == "exact" and request.context_length == 2048
    assert request.model == "model.gguf"
