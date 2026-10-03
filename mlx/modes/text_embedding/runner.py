from __future__ import annotations

from typing import Any

from mlx.core.commands import NullWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.core.configuration import with_explicit_options
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
    if config.get("pooling", "auto") != "auto" or config.get("prompt_format", "auto") != "auto":
        raise MLXUserError(
            "--pooling and --prompt-format require the BEIR workflow with --input; "
            "legacy CSV embedding does not support these options."
        )
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
            values["adapter"], device=str(values.get("device", "cpu")),
            trust_checkpoint_code=bool(values.get("trust_checkpoint_code", False)),
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


def _prepare_datasets(config):
    from mlx.modes.text_embedding.retrieval_datasets import PrepareRetrievalDatasets
    return PrepareRetrievalDatasets(config["dataset_path"], suite=config.get("suite", "laptop-ae-v1"), reporter=_reporter(config)).execute()


def _benchmark_autoencoders(config):
    from dataclasses import fields
    from mlx.modes.autoencoder.commands import TrainAutoencoder
    from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
    from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
    from mlx.modes.autoencoder.presentation import RichAutoencoderReporter
    from mlx.modes.text_embedding.experiment import BenchmarkAutoencoderRetrieval
    from mlx.modes.text_embedding.experiment_requests import AutoencoderRetrievalRequest

    defaults = AutoencoderRetrievalRequest()
    values = {field.name: getattr(defaults, field.name) for field in fields(defaults) if field.name != "extras"}
    explicit = config.get("_explicit_options", set())
    values.update({key: config[key] for key in values if key in explicit})
    for name in ("bottleneck_dims", "seeds", "k_values"):
        if isinstance(values[name], str):
            try:
                values[name] = tuple(int(item.strip()) for item in values[name].split(","))
            except ValueError as exc:
                raise MLXUserError(f"--{name.replace('_', '-')} requires comma-separated integers.") from exc
    for name in ("losses", "metrics"):
        if isinstance(values[name], str):
            values[name] = tuple(item.strip() for item in values[name].split(","))
    if values.get("experiment_config"):
        conflicts = set(explicit) & {"autoencoder_model", "losses", "bottleneck_dims", "similarity_weight", "seeds"}
        if conflicts:
            raise MLXUserError("--experiment-config conflicts with explicit variant flags: " + ", ".join(sorted(conflicts)))
    values["extras"] = {"explicit_options": sorted(explicit)}
    request = AutoencoderRetrievalRequest(**values)
    from mlx.modes.autoencoder.models import DEFAULT_AUTOENCODER_REGISTRY
    DEFAULT_AUTOENCODER_REGISTRY.resolve(request.autoencoder_model)
    training_reporter = NullWorkflowReporter() if config.get("output_format") == "json" else RichAutoencoderReporter()
    from mlx.modes.autoencoder.data import EmbeddingCsvLoader
    from mlx.modes.autoencoder.objectives import validate_training_variant
    return BenchmarkAutoencoderRetrieval(
        request, reporter=_reporter(config),
        training_factory=lambda options: TrainAutoencoder(AutoencoderTrainRequest(**options), reporter=training_reporter),
        transformer_factory=lambda path, **options: AutoencoderRepresentationTransformer(path, **{"device": request.device, **options}),
        vector_loader=EmbeddingCsvLoader(), variant_validator=validate_training_variant,
    ).execute()


def _select_autoencoder_settings(config):
    from mlx.modes.text_embedding.experiment_selection import SelectAutoencoderExperimentSettings
    from mlx.core.commands import emit
    if not config.get("input_path") or not config.get("output_path"):
        raise MLXUserError("Settings selection requires --input pilot-directory and --output new-directory.")
    result = SelectAutoencoderExperimentSettings(config["input_path"], config["output_path"]).execute()
    emit(_reporter(config), "success", f"Frozen confirmation config: {result['config']}", payload={"event": "retrieval_stage"})
    return result


ACTION_HANDLERS = {
    "embed": _embed, "benchmark": _benchmark,
    "select-autoencoder-settings": _select_autoencoder_settings,
    "prepare-datasets": _prepare_datasets, "benchmark-autoencoders": _benchmark_autoencoders,
    "ls-metrics": _list_components, "ls-vector-stores": _list_components,
    "ls-embedding-backends": _list_components,
}


def run_text_embedding(config: dict[str, Any]) -> Any:
    config = with_explicit_options(config)
    action = config.get("action") or "embed"
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTION_HANDLERS))
        raise MLXUserError(
            f"Unsupported action '{action}' for text_embedding. Available actions: {available}."
        )
    return handler(config)


__all__ = ["run_text_embedding"]
