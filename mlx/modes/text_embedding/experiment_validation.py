"""Shared validation for legacy and configured retrieval protocols."""
import math
from pathlib import Path
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.retrieval_datasets import validate_suite

def validate_experiment_request(r, datasets):
    validate_suite(r.suite)
    if not r.model or not Path(r.model).expanduser().is_file():
        raise MLXUserError("Provide an existing GGUF embedding model with --model.")
    if Path(r.model).suffix.lower() != ".gguf":
        raise MLXUserError("The embedding model must be a .gguf file.")
    from mlx.modes.text_embedding.embedding.registry import DEFAULT_EMBEDDING_BACKENDS
    from mlx.modes.text_embedding.formatting import resolve_text_formatter
    backend = DEFAULT_EMBEDDING_BACKENDS.resolve(r.embedding_backend)
    if not backend.supports_context_length:
        raise MLXUserError("Experiment embedding backend must support explicit context length.")
    resolve_text_formatter(r.prompt_format)
    if r.pooling not in ("auto", "mean", "cls", "last", "none"):
        raise MLXUserError("Unsupported experiment pooling choice.")
    if not r.output_path or not r.dataset_path:
        raise MLXUserError("Provide --output and --dataset-path for the experiment.")
    if len(datasets) < 2 or len(set(datasets)) != len(datasets):
        raise MLXUserError("Experiment requires at least two distinct datasets.")
    if not r.losses or len(set(r.losses)) != len(r.losses) or set(r.losses) - {"mse", "mse-similarity"}:
        raise MLXUserError("This ablation supports distinct losses mse,mse-similarity.")
    if (len(r.seeds) < 2 or len(set(r.seeds)) != len(r.seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in r.seeds)):
        raise MLXUserError("Provide at least two distinct seeds between 0 and 2**32-1.")
    if (not r.bottleneck_dims or len(set(r.bottleneck_dims)) != len(r.bottleneck_dims)
            or any(d < 1 or d > r.hidden_dim for d in r.bottleneck_dims)):
        raise MLXUserError("Bottleneck dimensions must be distinct, positive, and no larger than hidden-dim.")
    if r.vector_store != "exact" or not r.normalize_embeddings:
        raise MLXUserError("The experiment requires --vector-store exact and normalized embeddings.")
    if r.batch_size < 2 or r.embedding_batch_size < 1 or r.epochs < 1 or r.context_length < 1:
        raise MLXUserError("Require batch-size >= 2, positive embedding batch size, epochs, and context length.")
    if not 0 < r.val_ratio < 1 or not math.isfinite(r.lr) or r.lr <= 0:
        raise MLXUserError("Require 0 < val-ratio < 1 and a finite positive learning rate.")
    if not math.isfinite(r.similarity_weight) or r.similarity_weight < 0:
        raise MLXUserError("Similarity weight must be finite and nonnegative.")
    if not 0 < r.alpha < 1 or not 0 < r.noninferiority_margin < 1:
        raise MLXUserError("Alpha and non-inferiority margin must be between zero and one.")
    if not r.k_values or min(r.k_values) < 1 or max(r.k_values) > r.top_k:
        raise MLXUserError("Metric cutoffs must be positive and no larger than top-k.")
    from mlx.modes.text_embedding.metric_registry import DEFAULT_METRIC_REGISTRY
    for metric in r.metrics:
        DEFAULT_METRIC_REGISTRY.resolve(metric)
    if r.primary_metric != "ndcg@10" or "ndcg" not in r.metrics or 10 not in r.k_values:
        raise MLXUserError("The primary outcome is ndcg@10; include ndcg and cutoff 10.")
