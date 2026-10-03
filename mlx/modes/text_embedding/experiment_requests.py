from __future__ import annotations

from dataclasses import dataclass

from mlx.core.requests import ConfigRequest


@dataclass(frozen=True)
class AutoencoderRetrievalRequest(ConfigRequest):
    experiment_config: str | None = None
    embedding_source: str | None = None
    dry_run: bool = False
    dataset_path: str = ""
    output_path: str = ""
    model: str = ""
    suite: str = "laptop-ae-v1"
    download_datasets: bool = False
    resume: bool = False
    embedding_backend: str = "llama-cpp"
    prompt_format: str = "embeddinggemma"
    pooling: str = "auto"
    context_length: int = 2048
    embedding_batch_size: int = 1
    normalize_embeddings: bool = True
    vector_store: str = "exact"
    exclude_self_matches: bool = True
    autoencoder_model: str = "simple"
    losses: tuple[str, ...] = ("mse", "mse-similarity")
    similarity_weight: float = 1.0
    bottleneck_dims: tuple[int, ...] = (128, 256)
    hidden_dim: int = 256
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    val_ratio: float = 0.2
    primary_metric: str = "ndcg@10"
    noninferiority_margin: float = 0.01
    alpha: float = 0.05
    top_k: int = 100
    k_values: tuple[int, ...] = (10, 100)
    metrics: tuple[str, ...] = ("ndcg", "recall", "mrr", "map")
    device: str = "cpu"
