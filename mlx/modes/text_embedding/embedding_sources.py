"""Validate external embedding stages without modifying their source experiment."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.artifacts import EmbeddingArtifactReader
from mlx.modes.text_embedding.experiment_artifacts import file_hashes, read_json


class ValidateEmbeddingSource:
    def __init__(self, root, request, dataset_hashes, model_hash):
        self.root, self.request = Path(root).expanduser().resolve(), request
        self.hashes, self.model_hash = dataset_hashes, model_hash

    def execute(self):
        identity = read_json(self.root / "experiment.json")
        if identity.get("model_sha256") != self.model_hash:
            raise MLXUserError("Embedding source model hash differs from the requested GGUF.")
        source_config = identity.get("config", {})
        for key in ("embedding_backend", "prompt_format", "pooling", "context_length", "normalize_embeddings"):
            if source_config.get(key) != getattr(self.request, key):
                raise MLXUserError(f"Embedding source has incompatible {key}.")
        try:
            runtime = version("llama-cpp-python")
        except PackageNotFoundError:
            runtime = None
        if identity.get("packages", {}).get("llama-cpp-python") != runtime:
            raise MLXUserError("Embedding source llama-cpp-python version differs from the current runtime.")
        result = {}
        for name, hashes in self.hashes.items():
            if identity.get("datasets", {}).get(name) != hashes:
                raise MLXUserError(f"Embedding source dataset hashes differ for {name}.")
            path = self.root / name / "original"
            state = read_json(path / "stage.json")
            if not state.get("hashes") or state["hashes"] != file_hashes(path):
                raise MLXUserError(f"Embedding source stage is incomplete or modified: {path}.")
            artifact = EmbeddingArtifactReader().load(path)
            manifest = artifact["embedding_manifest"]
            if (manifest["embedding"].get("representation") != "original"
                    or not manifest["embedding"].get("normalized")
                    or manifest.get("model", {}).get("sha256") != self.model_hash):
                raise MLXUserError(f"Embedding source must contain original normalized vectors for {name}.")
            configuration = manifest.get("embedding_configuration", {})
            for field, expected in (("context_length", self.request.context_length),
                                    ("pooling_requested", self.request.pooling),
                                    ("prompt_format_effective", self.request.prompt_format),
                                    ("llama_cpp_python_version", runtime)):
                if field in configuration and configuration[field] != expected:
                    raise MLXUserError(f"Embedding source manifest has incompatible {field} for {name}.")
            if manifest.get("vector_store", {}).get("provider") != self.request.vector_store:
                raise MLXUserError(f"Embedding source vector store is incompatible for {name}.")
            result[name] = {"path": str(path), "hashes": state["hashes"],
                            "dimensions": manifest["embedding"]["dimensions"],
                            "embedding_configuration": manifest.get("embedding_configuration", {})}
        return result
