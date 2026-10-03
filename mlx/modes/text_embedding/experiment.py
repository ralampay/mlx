"""Coordinate a reproducible corpus-adapted compression experiment."""
from __future__ import annotations

from pathlib import Path

from mlx.core.artifacts import sha256_file
from mlx.core.commands import NullWorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.artifacts import EmbeddingArtifactReader
from mlx.modes.text_embedding.commands import EmbedTextCommand, BenchmarkTextEmbeddingCommand
from mlx.modes.text_embedding.data import BeirDatasetLoader
from mlx.modes.text_embedding.experiment_artifacts import ExperimentStages, experiment_identity
from mlx.modes.text_embedding.experiment_statistics import AnalyzeAutoencoderRetrieval, variant_name
from mlx.modes.text_embedding.requests import EmbedTextRequest, BenchmarkTextEmbeddingRequest
from mlx.modes.text_embedding.retrieval_datasets import (
    SUITE_DATASETS, PrepareRetrievalDatasets, dataset_hashes,
)
from mlx.modes.text_embedding.transforms import TransformEmbeddingArtifacts


class BenchmarkAutoencoderRetrieval:
    def __init__(self, request, *, training_factory, transformer_factory, reporter=None,
                 embed_factory=EmbedTextCommand, benchmark_factory=BenchmarkTextEmbeddingCommand,
                 datasets=SUITE_DATASETS, prepare_factory=PrepareRetrievalDatasets,
                 vector_loader=None, variant_validator=None):
        self.vector_loader = vector_loader
        self.variant_validator = variant_validator
        self.request = request
        self.training_factory = training_factory
        self.transformer_factory = transformer_factory
        self.reporter = reporter or NullWorkflowReporter()
        self.embed_factory = embed_factory
        self.benchmark_factory = benchmark_factory
        self.datasets = tuple(datasets)
        self.prepare_factory = prepare_factory

    def execute(self):
        if self.request.experiment_config:
            from mlx.modes.text_embedding.configured_experiment import BenchmarkConfiguredAutoencoders
            return BenchmarkConfiguredAutoencoders(
                self.request, training_factory=self.training_factory, transformer_factory=self.transformer_factory,
                vector_loader=self.vector_loader, variant_validator=self.variant_validator,
                reporter=self.reporter, embed_factory=self.embed_factory,
                benchmark_factory=self.benchmark_factory, prepare_factory=self.prepare_factory,
            ).execute()
        self._validate()
        if self.request.dry_run:
            return {"datasets": len(self.datasets), "training_runs": len(self.datasets) * len(self.request.losses) * len(self.request.bottleneck_dims) * len(self.request.seeds)}
        r = self.request
        root = Path(r.dataset_path).expanduser()
        if r.download_datasets:
            self.prepare_factory(root, suite=r.suite, reporter=self.reporter).execute()
        hashes = {}
        for name in self.datasets:
            BeirDatasetLoader(allow_empty_documents=True).load(root / name)
            hashes[name] = dataset_hashes(root / name)
            provenance = root / name / "source_manifest.json"
            if provenance.exists():
                hashes[name]["source_manifest.json"] = sha256_file(provenance)
        identity = experiment_identity(r, sha256_file(Path(r.model).expanduser()), hashes)
        stages = ExperimentStages(Path(r.output_path).expanduser(), identity, resume=r.resume, reporter=self.reporter)
        for name in self.datasets:
            self._dataset(name, root / name, stages)
        statistics = stages.run("comparison", lambda output: AnalyzeAutoencoderRetrieval(
            r, self.datasets, stages.root, output).execute())
        emit(self.reporter, "success", f"Experiment complete: {statistics / 'report.md'}",
             payload={"event": "retrieval_stage"})
        return {"output_dir": str(stages.root), "report": str(statistics / "report.md"),
                "datasets": len(self.datasets), "training_runs": len(self.datasets) * len(r.losses) * len(r.bottleneck_dims) * len(r.seeds)}

    def _dataset(self, name, source, stages):
        r = self.request
        original = stages.run(f"{name}/original", lambda output: self.embed_factory(
            EmbedTextRequest(model=r.model, input_path=str(source), output_path=str(output),
                             vector_store=r.vector_store, embedding_backend=r.embedding_backend,
                             prompt_format=r.prompt_format, pooling=r.pooling, context_length=r.context_length,
                             batch_size=r.embedding_batch_size, normalize_embeddings=True),
            reporter=self.reporter, dataset_loader=BeirDatasetLoader(allow_empty_documents=True)).execute())
        dimensions = EmbeddingArtifactReader().load(original)["embedding_manifest"]["embedding"]["dimensions"]
        if max(r.bottleneck_dims) >= dimensions:
            raise MLXUserError(f"Bottleneck dimensions must be smaller than the model output ({dimensions}).")
        stages.run(f"{name}/baseline", lambda output: self._benchmark(original, output))
        for dimension in r.bottleneck_dims:
            for seed in r.seeds:
                for loss in r.losses:
                    cell = f"{name}/{variant_name(loss, dimension, seed)}"
                    training = stages.run(f"{cell}/training", lambda output: self.training_factory({
                        "model": r.autoencoder_model, "input_path": str(original / "corpus_embeddings.csv"),
                        "output_path": str(output), "hidden_dim": r.hidden_dim, "bottleneck_dim": dimension,
                        "loss": loss, "loss_config": {"similarity_weight": r.similarity_weight} if loss == "mse-similarity" else {},
                        "random_seed": seed, "epochs": r.epochs, "batch_size": r.batch_size,
                        "lr": r.lr, "val_ratio": r.val_ratio, "device": r.device,
                        "minimum_batch_size": 2, "plots": False,
                    }).execute())
                    def transform(output):
                        adapter = self.transformer_factory(training / "autoencoder.pth")
                        return TransformEmbeddingArtifacts(original, output, transformer=adapter,
                                                           vector_store=r.vector_store, batch_size=r.batch_size).execute()
                    embeddings = stages.run(f"{cell}/embeddings", transform)
                    stages.run(f"{cell}/benchmark", lambda output: self._benchmark(embeddings, output))

    def _benchmark(self, source, output):
        r = self.request
        return self.benchmark_factory(BenchmarkTextEmbeddingRequest(
            input_path=str(source), output_path=str(output), vector_store=r.vector_store,
            top_k=r.top_k, k_values=r.k_values, metrics=r.metrics,
            exclude_self_matches=r.exclude_self_matches), reporter=self.reporter).execute()

    def _validate(self):
        from mlx.modes.text_embedding.experiment_validation import validate_experiment_request
        validate_experiment_request(self.request, self.datasets)
