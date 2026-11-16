"""Composition and presentation only; commands remain callable without CLI state."""

from mlx.core.commands import NullWorkflowReporter
from mlx.core.configuration import load_component_options, with_explicit_options
from mlx.core.exceptions import MLXUserError

from .commands import (
    BenchmarkSyntheticTimeSeries,
    SynthesizeTimeSeries,
    TrainTimeSeriesGenerator,
)
from .presentation import RichTemporalReporter, display_result
from .registry import TemporalModelRegistry
from .requests import BenchmarkRequest, SynthesizeRequest, TrainRequest


def run_time_series(config):
    values = with_explicit_options(config)
    explicit = values["_explicit_options"]
    for key, default in {
        "model": "timevae",
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.001,
        "random_seed": 42,
        "device": "cpu",
        "bootstrap_samples": 100,
    }.items():
        if key not in explicit:
            values[key] = default
    action = values.get("action") or "ls-models"
    if action == "ls-models":
        result = TemporalModelRegistry().inventory()
    elif action == "train":
        columns = values.get("feature_columns") or ()
        widths = values.get("hidden_dims", (64, 128))
        values["feature_columns"] = (
            tuple(c.strip() for c in columns.split(","))
            if isinstance(columns, str)
            else tuple(columns)
        )
        try:
            values["hidden_dims"] = (
                tuple(int(c.strip()) for c in widths.split(","))
                if isinstance(widths, str)
                else tuple(widths)
            )
        except (TypeError, ValueError) as exc:
            raise MLXUserError(
                "hidden-dims must be comma-separated positive integer widths."
            ) from exc
        values["model_config"] = load_component_options(
            values.get("model_config"), purpose="temporal model"
        )
        reporter = (
            NullWorkflowReporter()
            if values.get("output_format") == "json"
            else RichTemporalReporter()
        )
        result = TrainTimeSeriesGenerator(
            TrainRequest.from_config(values), reporter=reporter
        ).execute()
    elif action == "synthesize":
        if "random_seed" not in explicit:
            values["random_seed"] = 123
        result = SynthesizeTimeSeries(SynthesizeRequest.from_config(values)).execute()
    elif action == "benchmark":
        result = BenchmarkSyntheticTimeSeries(
            BenchmarkRequest.from_config(values)
        ).execute()
    else:
        raise MLXUserError(
            f"Unsupported time_series action '{action}'; use ls-models, train, synthesize, or benchmark."
        )
    if values.get("output_format") != "json":
        display_result(result)
    return result
