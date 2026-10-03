"""Coordinate named compression variants using injected training and vector adapters."""
from dataclasses import asdict, replace
from pathlib import Path

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.commands import emit
from mlx.core.exceptions import MLXUserError
from mlx.core.partitions import partition_rows, partition_hash
from mlx.modes.text_embedding.compression_controls import FitLinearProjection, LinearProjectionVectors, TruncateVectors
from mlx.modes.text_embedding.data import BeirDatasetLoader
from mlx.modes.text_embedding.embedding_sources import ValidateEmbeddingSource
from mlx.modes.text_embedding.experiment_artifacts import ExperimentStages, experiment_identity, read_json
from mlx.modes.text_embedding.experiment_config import load_experiment_config, run_name
from mlx.modes.text_embedding.requests import EmbedTextRequest, BenchmarkTextEmbeddingRequest
from mlx.modes.text_embedding.retrieval_datasets import dataset_hashes
from mlx.modes.text_embedding.transforms import TransformEmbeddingArtifacts


class BenchmarkConfiguredAutoencoders:
    def __init__(self, request, *, training_factory, transformer_factory, vector_loader,
                 variant_validator, reporter, embed_factory, benchmark_factory, prepare_factory):
        config_path = str(Path(request.experiment_config).expanduser().resolve())
        self.config = load_experiment_config(config_path)
        self.request = self.config.resolve_request(replace(request, experiment_config=config_path))
        self.training_factory, self.transformer_factory = training_factory, transformer_factory
        self.vector_loader, self.variant_validator = vector_loader, variant_validator
        self.reporter, self.embed_factory = reporter, embed_factory
        self.benchmark_factory, self.prepare_factory = benchmark_factory, prepare_factory

    def execute(self):
        r, c = self.request, self.config
        if self.vector_loader is None or self.variant_validator is None:
            raise MLXUserError("Configured experiments require injected vector_loader and variant_validator collaborators.")
        # Reuse existing protocol validation without introducing another backend dispatch.
        from mlx.modes.text_embedding.experiment_validation import validate_experiment_request
        validation_request = replace(r, experiment_config=None, losses=("mse",), bottleneck_dims=(1,))
        validate_experiment_request(validation_request, c.datasets)
        root = Path(r.dataset_path).expanduser()
        if r.download_datasets and not r.dry_run:
            self.prepare_factory(root, suite=r.suite, reporter=self.reporter).execute()
        hashes, row_counts = {}, {}
        for name in c.datasets:
            dataset = BeirDatasetLoader(allow_empty_documents=True).load(root / name)
            row_counts[name] = len(dataset.corpus)
            hashes[name] = dataset_hashes(root / name)
            if (root / name / "source_manifest.json").exists():
                hashes[name]["source_manifest.json"] = sha256_file(root / name / "source_manifest.json")
        model_hash = sha256_file(Path(r.model).expanduser())
        sources = ValidateEmbeddingSource(r.embedding_source, r, hashes, model_hash).execute() if r.embedding_source else {}
        for v in c.variants:
            for width in v.dimensions:
                minimum_rows = width if v.kind in ("pca", "svd") else 1
                if v.kind == "autoencoder":
                    minimum_rows = self.variant_validator(v, r, width) or 1
                if v.kind in ("autoencoder", "pca", "svd"):
                    for count in row_counts.values():
                        validation_rows = min(count - 1, max(1, round(count * r.val_ratio)))
                        if v.kind == "autoencoder" and min(validation_rows, count - validation_rows) < 2:
                            raise MLXUserError("Experiment training and validation partitions need at least two document rows.")
                        if minimum_rows > count - validation_rows:
                            raise MLXUserError("SVD-based fitting requires at least as many training documents as components.")
                if sources and any(width >= source["dimensions"] for source in sources.values()):
                    raise MLXUserError(f"{v.name} dimensions must be smaller than the source embeddings.")
        counts = c.counts()
        emit(self.reporter, "info", f"{c.phase}: {counts}", payload={"event": "retrieval_stage"})
        if r.dry_run:
            return {"phase": c.phase, "dry_run": True, **counts, "variants": [asdict(v) for v in c.variants]}
        identity = experiment_identity(r, model_hash, hashes)
        identity.update(schema_version=2, experiment=asdict(c), embedding_sources=sources,
                        experiment_config_sha256=sha256_file(Path(r.experiment_config).expanduser()))
        stages = ExperimentStages(Path(r.output_path).expanduser(), identity, resume=r.resume, reporter=self.reporter)
        cells = []
        for name in c.datasets:
            original = self._original(name, root / name, sources, stages)
            cells.extend(self._dataset(name, original, stages))
        if c.phase == "pilot":
            path = stages.run("pilot", lambda out: write_json_atomic(out / "results.json", {
                "phase": "pilot", "config": asdict(c), "runtime": asdict(r), "cells": cells,
            }))
            result = {"pilot": str(path / "results.json")}
        else:
            from mlx.modes.text_embedding.configured_statistics import AnalyzeConfiguredAutoencoders
            path = stages.run("comparison", lambda out: AnalyzeConfiguredAutoencoders(r, c, stages.root, cells, out).execute())
            result = {"report": str(path / "report.md")}
        emit(self.reporter, "success", f"{c.phase} complete: {path}", payload={"event": "retrieval_stage"})
        return {"output_dir": str(stages.root), **counts, **result}

    def _original(self, name, dataset, sources, stages):
        if name in sources:
            stages.run(f"{name}/source", lambda out: write_json_atomic(out / "source.json", sources[name]))
            return Path(sources[name]["path"])
        r = self.request
        return stages.run(f"{name}/original", lambda out: self.embed_factory(
            EmbedTextRequest(model=r.model, input_path=str(dataset), output_path=str(out),
                             vector_store=r.vector_store, embedding_backend=r.embedding_backend,
                             prompt_format=r.prompt_format, pooling=r.pooling, context_length=r.context_length,
                             batch_size=r.embedding_batch_size, normalize_embeddings=True),
            reporter=self.reporter, dataset_loader=BeirDatasetLoader(allow_empty_documents=True)).execute())

    def _dataset(self, name, original, stages):
        r, c = self.request, self.config
        table = self.vector_loader.load(original / "corpus_embeddings.csv")
        if c.phase == "confirmation":
            stages.run(f"{name}/baseline", lambda out: self._benchmark(original, out))
        cells = []
        for variant in c.variants:
            for seed, width, evaluation in variant.runs(c.seeds):
                if width >= table.dimensions:
                    raise MLXUserError(f"{variant.name} must reduce the source embedding dimension.")
                base = f"{name}/runs/{run_name(variant, width, seed)}"
                training, metadata = None, {}
                if seed is not None:
                    rows, validation = partition_rows(len(table.vectors), r.val_ratio, seed)
                    split = partition_hash(rows, validation)
                    if variant.kind == "autoencoder":
                        training = stages.run(base + "/training", lambda out: self.training_factory({
                            "model": variant.model, "input_path": str(original / "corpus_embeddings.csv"),
                            "output_path": str(out), "hidden_dim": r.hidden_dim, "bottleneck_dim": width,
                            "autoencoder_config": variant.model_config, "loss": variant.loss,
                            "loss_config": variant.loss_config, "random_seed": seed, "epochs": r.epochs,
                            "batch_size": r.batch_size, "lr": r.lr, "val_ratio": r.val_ratio,
                            "device": r.device, "minimum_batch_size": 2, "plots": False,
                        }).execute())
                        metadata = read_json(training / "autoencoder_manifest.json")
                        if metadata.get("split_hash") != split:
                            raise MLXUserError("Training split differs from the declared experiment partition.")
                    elif variant.kind in ("pca", "svd"):
                        training = stages.run(base + "/training", lambda out: FitLinearProjection(
                            table.vectors, rows, width, out, split_hash=split, centered=variant.kind == "pca").execute())
                for dimension in evaluation:
                    cell = {"dataset": name, "variant": variant.name, "kind": variant.kind,
                            "dimension": dimension, "training_dimension": width, "seed": seed,
                            "run": base, "training": str(training.relative_to(stages.root)) if training else None,
                            "reconstruction": metadata.get("validation_components", {}).get("reconstruction"),
                            "split_hash": metadata.get("split_hash"), "original": str(original)}
                    if c.phase == "confirmation":
                        if variant.kind == "autoencoder":
                            adapter = self.transformer_factory(training / "autoencoder.pth", output_dimensions=dimension, device=r.device)
                        elif variant.kind in ("pca", "svd"):
                            adapter = LinearProjectionVectors(training / f"{variant.kind}.npz", dimension, kind=variant.kind)
                        else:
                            adapter = TruncateVectors(table.dimensions, dimension)
                        destination = base + f"/eval-{dimension}"
                        transformed = stages.run(destination + "/embeddings", lambda out: TransformEmbeddingArtifacts(
                            original, out, transformer=adapter, vector_store=r.vector_store,
                            batch_size=r.batch_size, representation=f"{variant.name}-{dimension}").execute())
                        stages.run(destination + "/benchmark", lambda out: self._benchmark(transformed, out))
                        cell["evaluation"] = destination
                    cells.append(cell)
        return cells

    def _benchmark(self, source, output):
        r = self.request
        return self.benchmark_factory(BenchmarkTextEmbeddingRequest(
            input_path=str(source), output_path=str(output), vector_store=r.vector_store,
            top_k=r.top_k, k_values=r.k_values, metrics=r.metrics,
            exclude_self_matches=r.exclude_self_matches), reporter=self.reporter).execute()
