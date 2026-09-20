from __future__ import annotations

from typing import Any

from mlx.core.commands import NullWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.commands import (
    BenchmarkTextEmbeddingCommand,
    EmbedTextCommand,
)
from mlx.modes.text_embedding.presentation import RichTextEmbeddingReporter
from mlx.modes.text_embedding.requests import (
    BenchmarkTextEmbeddingRequest,
    DEFAULT_K_VALUES,
    EmbedTextRequest,
)


def _reporter(config):
    return (
        NullWorkflowReporter()
        if config.get("output_format") == "json"
        else RichTextEmbeddingReporter()
    )


def _legacy_csv_embed(config: dict[str, Any]):
    from mlx.modes.nlp.embedding import EmbedCsvCommand, EmbedCsvRequest
    from mlx.modes.nlp.presentation import RichEmbeddingReporter

    is_json = config.get("output_format") == "json"
    return EmbedCsvCommand(
        EmbedCsvRequest.from_config({**config, "present": not is_json}),
        reporter=NullWorkflowReporter() if is_json else RichEmbeddingReporter(),
    ).execute()


def _embed(config: dict[str, Any]):
    if config.get("input_path") is None and config.get("input_file"):
        return _legacy_csv_embed(config)
    values = dict(config)
    explicit = config.get("_explicit_options", set())
    if "batch_size" not in explicit:
        values["batch_size"] = 16
    if not values.get("vector_store"):
        values["vector_store"] = "chroma"
    transformer = None
    if values.get("adapter"):
        from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer

        transformer = AutoencoderRepresentationTransformer(
            values["adapter"], device=str(values.get("device", "cpu"))
        )
    if not values.get("representation"):
        values["representation"] = (
            f"autoencoder-{transformer.output_dimensions}" if transformer else "original"
        )
    return EmbedTextCommand(
        EmbedTextRequest.from_config(values),
        reporter=_reporter(config),
        transformer=transformer,
    ).execute()


def _benchmark(config: dict[str, Any]):
    values = _parse_k_values(config.get("k_values", DEFAULT_K_VALUES))
    from mlx.modes.text_embedding.metric_registry import DEFAULT_METRICS
    metrics = config.get("metrics") or DEFAULT_METRICS
    if isinstance(metrics, str):
        metrics = tuple(name.strip().lower() for name in metrics.split(",") if name.strip())
    request = BenchmarkTextEmbeddingRequest.from_config({**config, "k_values": values, "metrics": tuple(metrics)})
    return BenchmarkTextEmbeddingCommand(request, reporter=_reporter(config)).execute()


def _parse_k_values(value) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise MLXUserError("--k-values must be a comma-separated list of integers.") from exc
    else:
        values = tuple(int(item) for item in value)
    return tuple(sorted(set(values)))


def _list_components(config):
    from mlx.core.model_listing import ListComponentNames
    from mlx.core.presentation import display_component_inventory
    from mlx.modes.text_embedding.metric_registry import DEFAULT_METRIC_REGISTRY
    from mlx.modes.text_embedding.vector_store.registry import DEFAULT_VECTOR_STORE_REGISTRY
    from mlx.modes.text_embedding.embedding.registry import DEFAULT_EMBEDDING_BACKENDS
    inventories = {
        "ls-metrics": DEFAULT_METRIC_REGISTRY.entries,
        "ls-vector-stores": DEFAULT_VECTOR_STORE_REGISTRY.names,
        "ls-embedding-backends": DEFAULT_EMBEDDING_BACKENDS.entries,
    }
    result = ListComponentNames(inventories[config["action"]]).execute()
    if config.get("output_format") != "json":
        display_component_inventory(result)
    return result


ACTION_HANDLERS = {
    "embed": _embed, "benchmark": _benchmark,
    "ls-metrics": _list_components, "ls-vector-stores": _list_components,
    "ls-embedding-backends": _list_components,
}


def run_text_embedding(config: dict[str, Any]) -> Any:
    action = config.get("action") or "embed"
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for text_embedding. Available actions: {available}."
        )
    return handler(config)


__all__ = ["run_text_embedding"]
