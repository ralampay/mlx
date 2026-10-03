"""Transform and reindex saved embeddings without rerunning their text model."""
from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.artifacts import EmbeddingArtifactReader, prepare_new_output_directory
from mlx.modes.text_embedding.vector_store.protocol import VectorRecord
from mlx.modes.text_embedding.vector_store.registry import DEFAULT_VECTOR_STORE_REGISTRY


class TransformEmbeddingArtifacts:
    def __init__(self, input_path, output_path, *, transformer, vector_store="exact", batch_size=64,
                 registry=DEFAULT_VECTOR_STORE_REGISTRY, representation=None):
        self.input = Path(input_path)
        self.output = Path(output_path)
        self.transformer = transformer
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.registry = registry
        self.representation = representation or f"{transformer.provenance.get('type', 'transform')}-{transformer.output_dimensions}"

    def execute(self):
        artifacts = EmbeddingArtifactReader().load(self.input)
        manifest = deepcopy(artifacts["embedding_manifest"])
        if self.batch_size < 1 or manifest["embedding"]["dimensions"] != self.transformer.input_dimensions:
            raise MLXUserError("Transform requires a positive batch size and matching source dimensions.")
        factory = self.registry.resolve(self.vector_store)
        output = prepare_new_output_directory(self.output, purpose="Embedding transform")
        store = factory(output / "vector_store", collection="corpus", create=True)
        try:
            for kind in ("corpus", "query"):
                self._transform_csv(kind, store)
        finally:
            store.close()
        shutil.copyfile(self.input / "dataset_manifest.json", output / "dataset_manifest.json")
        qrels_name = artifacts["dataset_manifest"].get("qrels_path", "qrels.tsv")
        (output / qrels_name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input / qrels_name, output / qrels_name)
        manifest["embedding"].update(dimensions=self.transformer.output_dimensions, normalized=True,
                                     representation=self.representation)
        manifest["adapter"] = dict(self.transformer.provenance)
        manifest["vector_store"] = {"provider": self.vector_store, "similarity": "cosine"}
        write_json_atomic(output / "embedding_manifest.json", manifest)
        EmbeddingArtifactReader().load(output)
        return {"output_dir": str(output), "dimensions": self.transformer.output_dimensions}

    def _transform_csv(self, kind, store):
        filename = f"{kind}_embeddings.csv"
        with (self.input / filename).open(newline="", encoding="utf-8") as source, (self.output / filename).open("w", newline="", encoding="utf-8") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) == self.batch_size:
                    self._write_batch(batch, kind, writer, store)
                    batch = []
            if batch:
                self._write_batch(batch, kind, writer, store)

    def _write_batch(self, rows, kind, writer, store):
        vectors = [json.loads(row["embedding"]) for row in rows]
        values = np.asarray(self.transformer.transform(vectors), dtype=np.float32)
        if values.shape != (len(rows), self.transformer.output_dimensions) or not np.isfinite(values).all():
            raise MLXUserError("Representation transformer returned invalid vectors.")
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        for row, vector in zip(rows, values.tolist()):
            row["embedding"] = json.dumps(vector, separators=(",", ":"))
            writer.writerow(row)
        if kind == "corpus":
            store.add([VectorRecord(row["id"], vector, row["text"]) for row, vector in zip(rows, values.tolist())])
