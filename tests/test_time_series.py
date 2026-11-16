"""Portable temporal-generation contracts using small synthetic series."""

import csv
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.time_series.artifacts import TemporalCheckpointStore, output_directory
from mlx.modes.time_series.commands import (
    BenchmarkSyntheticTimeSeries,
    SynthesizeTimeSeries,
    TrainTimeSeriesGenerator,
)
from mlx.modes.time_series.data import TemporalCsvLoader, load_synthetic
from mlx.modes.time_series.metrics import distribution_metrics, nearest_distances
from mlx.modes.time_series.registry import TemporalModelRegistry
from mlx.modes.time_series.requests import (
    BenchmarkRequest,
    SynthesizeRequest,
    TrainRequest,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_series(path, count=160, groups=1):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "signal", "other", "id"])
        for group in range(groups):
            for step in reversed(range(count)):
                writer.writerow(
                    [step, np.sin(step / 4) + 0.01 * step, np.cos(step / 4), str(group)]
                )
    return path


def request(tmp_path, model="vanilla-vae"):
    return TrainRequest(
        input_path=str(write_series(tmp_path / "data.csv")),
        output_path=str(tmp_path / model),
        time_column="time",
        feature_columns=("signal", "other"),
        window_size=8,
        stride=4,
        hidden_dims=(4, 8),
        latent_dim=3,
        epochs=2,
        batch_size=8,
        model=model,
    )


def test_split_before_window_and_training_scaler(tmp_path):
    path = write_series(tmp_path / "data.csv", groups=2)
    prepared = TemporalCsvLoader().load(
        path,
        time_column="time",
        feature_columns=("other", "signal"),
        series_id_column="id",
        window_size=8,
        stride=2,
    )
    assert prepared.metadata["feature_columns"] == ["other", "signal"]
    expected = np.concatenate(list(prepared.partitions["train"].values()))
    np.testing.assert_allclose(prepared.metadata["mean"], expected.mean(0))
    for split in ("train", "val", "test"):
        windows, groups = prepared.windows(split, standardized=False)
        assert set(groups) == {"0", "1"}
        for values in prepared.partitions[split].values():
            assert all(
                any(
                    np.allclose(window, values[i : i + 8])
                    for i in range(0, len(values) - 7, 2)
                )
                for window in windows
            )


@pytest.mark.parametrize("model", ["vanilla-vae", "conv-vae", "timevae"])
def test_train_reload_synthesize_cpu(tmp_path, model):
    training = request(tmp_path, model)
    if model == "timevae":
        training = replace(
            training, model_config={"seasonal_periods": [4], "trend_degree": 2}
        )
    original_rng = torch.random.get_rng_state().clone()
    result = TrainTimeSeriesGenerator(training).execute()
    assert torch.equal(original_rng, torch.random.get_rng_state())
    checkpoint = TemporalCheckpointStore().load(
        result["checkpoint"], TemporalModelRegistry()
    )
    assert len(json.loads((tmp_path / model / "history.json").read_text())) == 2
    assert checkpoint["preprocessing"]["window_size"] == 8
    (tmp_path / "data.csv").unlink()  # Sampling must not require source data.
    generation = SynthesizeRequest(
        model_path=result["checkpoint"],
        output_path=str(tmp_path / "generated"),
        num_sequences=5,
        batch_size=2,
    )
    output = SynthesizeTimeSeries(generation).execute()
    values = load_synthetic(output["synthetic_csv"], checkpoint["preprocessing"])
    assert values.shape == (5, 8, 2)
    assert np.isfinite(values).all()
    second = SynthesizeTimeSeries(
        replace(generation, output_path=str(tmp_path / "again"))
    ).execute()
    np.testing.assert_array_equal(
        values, load_synthetic(second["synthetic_csv"], checkpoint["preprocessing"])
    )
    assert torch.equal(original_rng, torch.random.get_rng_state())


def test_benchmark_and_report(tmp_path):
    training = request(tmp_path)
    result = TrainTimeSeriesGenerator(training).execute()
    generated = SynthesizeTimeSeries(
        SynthesizeRequest(
            model_path=result["checkpoint"],
            output_path=str(tmp_path / "generated"),
            num_sequences=8,
        )
    ).execute()
    report = BenchmarkSyntheticTimeSeries(
        BenchmarkRequest(
            model_path=result["checkpoint"],
            input_path=training.input_path,
            synthetic_data=generated["synthetic_csv"],
            output_path=str(tmp_path / "quality"),
            bootstrap_samples=3,
        )
    ).execute()
    metrics = report["metrics"]
    assert metrics["protocol"]["evaluated_window_counts"]["real"] == 3
    assert metrics["fidelity"]["per_feature"].keys() == {"signal", "other"}
    assert (tmp_path / "quality" / "report.html").exists()
    assert (tmp_path / "quality" / "feature-1.png").exists()
    assert json.loads((tmp_path / "quality" / "metrics.json").read_text()) == metrics


def test_metrics_detect_shift_shuffle_and_duplicates():
    rng = np.random.default_rng(4)
    values = np.cumsum(rng.normal(size=(20, 32, 2)), axis=1)
    same = distribution_metrics(values, values, ["a", "b"])["aggregate"]
    assert same["wasserstein"] == 0
    assert same["acf_mae"] == 0
    assert same["std_ratio"] == 1
    shifted = distribution_metrics(values, values + 5, ["a", "b"])["aggregate"]
    assert shifted["wasserstein"] == pytest.approx(5)
    shuffled = distribution_metrics(values, values[:, rng.permutation(32)], ["a", "b"])[
        "aggregate"
    ]
    assert shuffled["wasserstein"] == 0
    assert shuffled["acf_mae"] > 0.05
    assert nearest_distances(values, values)["mean"] == pytest.approx(0, abs=1e-7)
    constant = distribution_metrics(np.ones((2, 8, 1)), np.ones((2, 8, 1)), ["a"])
    assert constant["aggregate"]["acf_mae"] is None
    assert constant["aggregate"]["std_ratio"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows.__setitem__(1, rows[0].copy()), "distinct"),
        (lambda rows: rows[0].__setitem__(0, "999"), "irregular"),
        (lambda rows: rows[0].__setitem__(1, ""), "Non-numeric"),
        (lambda rows: rows[0].__setitem__(1, "nan"), "Non-finite"),
    ],
)
def test_csv_errors(tmp_path, mutation, match):
    path = write_series(tmp_path / "data.csv")
    with path.open() as stream:
        header, *rows = list(csv.reader(stream))
    mutation(rows)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    with pytest.raises(MLXUserError, match=match):
        TemporalCsvLoader().load(
            path, time_column="time", feature_columns=("signal",), window_size=8
        )


def test_insufficient_data_and_invalid_columns(tmp_path):
    path = write_series(tmp_path / "data.csv", count=10)
    with pytest.raises(MLXUserError, match="needs window-size"):
        TemporalCsvLoader().load(
            path, time_column="time", feature_columns=("signal",), window_size=8
        )
    with pytest.raises(MLXUserError, match="distinct and present"):
        TemporalCsvLoader().load(
            path, time_column="time", feature_columns=("time",), window_size=8
        )


def test_corrupt_checkpoint_and_reference_change(tmp_path):
    training = request(tmp_path)
    result = TrainTimeSeriesGenerator(training).execute()
    store = TemporalCheckpointStore()
    registry = TemporalModelRegistry()
    checkpoint = store.load(result["checkpoint"], registry)
    checkpoint["definition"] = "untrusted.module:Class"
    store.save(tmp_path / "bad.pt", checkpoint)
    with pytest.raises(MLXUserError, match="not trusted"):
        store.load(tmp_path / "bad.pt", registry)
    checkpoint = store.load(result["checkpoint"], registry)
    with open(training.input_path, "a") as stream:
        stream.write("\n")
    with pytest.raises(MLXUserError, match="differs"):
        TemporalCsvLoader().load(
            training.input_path,
            time_column="time",
            feature_columns=("signal", "other"),
            window_size=8,
            stride=4,
            expected=checkpoint["preprocessing"],
        )


def test_output_collision_and_failure_cleanup(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    with pytest.raises(MLXUserError, match="already exists"), output_directory(target):
        pytest.fail("must not start work")
    other = tmp_path / "failure"
    with pytest.raises(ValueError), output_directory(other) as staging:
        (staging / "partial.txt").write_text("partial")
        raise ValueError("test")
    assert not other.exists()
    assert not list(tmp_path.glob(".failure-*"))


def test_cli_mode_and_json_inventory():
    from mlx.cli import build_parser
    from mlx.cli_routing import resolve_mode_descriptor

    args = build_parser().parse_args(
        ["--mode", "time-series", "--feature-columns", "a,b", "--latent-dim", "3"]
    )
    assert args.latent_dim == 3
    assert resolve_mode_descriptor("time-series").default_action == "ls-models"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx",
            "--mode",
            "time-series",
            "--action",
            "ls-models",
            "--format",
            "json",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert {row["name"] for row in json.loads(completed.stdout)} == {
        "vanilla-vae",
        "conv-vae",
        "timevae",
    }


def test_iso_millisecond_timestamps_and_univariate(tmp_path):
    from datetime import datetime, timedelta, timezone

    path = tmp_path / "dates.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "value"])
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(100):
            writer.writerow([(start + timedelta(milliseconds=i)).isoformat(), i])
    data = TemporalCsvLoader().load(
        path, time_column="date", feature_columns=("value",), window_size=8
    )
    assert data.metadata["time_unit"] == "seconds"
    assert data.metadata["sampling_interval"] == pytest.approx(0.001, abs=1e-6)
    assert data.windows("test")[0].shape == (1, 8, 1)


def test_synthesis_inverse_scaling_with_injected_backend(tmp_path):
    training = request(tmp_path)
    result = TrainTimeSeriesGenerator(training).execute()
    checkpoint = TemporalCheckpointStore().load(
        result["checkpoint"], TemporalModelRegistry()
    )

    class ZeroLatentDecoder:
        def sample(self, definition, checkpoint, request):
            yield np.zeros((request.num_sequences, 8, 2))

    generated = SynthesizeTimeSeries(
        SynthesizeRequest(
            model_path=result["checkpoint"],
            output_path=str(tmp_path / "generated"),
            num_sequences=2,
        ),
        backend=ZeroLatentDecoder(),
    ).execute()
    values = load_synthetic(generated["synthetic_csv"], checkpoint["preprocessing"])
    np.testing.assert_allclose(
        values, np.broadcast_to(checkpoint["preprocessing"]["mean"], values.shape)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("lr", float("nan")),
        ("kl_weight", -1),
        ("window_size", 1),
        ("model_config", []),
        ("hidden_dims", (0,)),
    ],
)
def test_invalid_training_settings(tmp_path, field, value):
    with pytest.raises(MLXUserError):
        TrainTimeSeriesGenerator(replace(request(tmp_path), **{field: value})).execute()
    assert not (tmp_path / "vanilla-vae").exists()


def test_runner_defaults_and_explicit_options(monkeypatch):
    from mlx.modes.time_series import runner

    captured = []

    class Capture:
        def __init__(self, request, **kwargs):
            captured.append(request)

        def execute(self):
            return {}

    monkeypatch.setattr(runner, "TrainTimeSeriesGenerator", Capture)
    runner.run_time_series(
        {
            "action": "train",
            "model": "other-mode-default",
            "epochs": 9,
            "feature_columns": "a,b",
            "hidden_dims": "4,8",
            "output_format": "json",
            "_explicit_options": {"feature_columns", "hidden_dims"},
        }
    )
    assert captured[-1].model == "timevae"
    assert captured[-1].epochs == 100
    assert captured[-1].feature_columns == ("a", "b")
    runner.run_time_series({"action": "train", "epochs": 3, "output_format": "json"})
    assert captured[-1].epochs == 3


def test_cli_train_synthesize_benchmark(tmp_path):
    import os

    path = write_series(tmp_path / "data.csv")
    runtime = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )

    def run(*args):
        command = [
            sys.executable,
            "-m",
            "mlx",
            "--mode",
            "time_series",
            "--format",
            "json",
            *map(str, args),
        ]
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, env=runtime
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    trained = run(
        "--action",
        "train",
        "--input",
        path,
        "--time-column",
        "time",
        "--feature-columns",
        "signal",
        "--model",
        "conv-vae",
        "--epochs",
        1,
        "--window-size",
        8,
        "--stride",
        4,
        "--hidden-dims",
        "4",
        "--output",
        tmp_path / "trained",
    )
    generated = run(
        "--action",
        "synthesize",
        "--model-path",
        trained["checkpoint"],
        "--num-sequences",
        3,
        "--output",
        tmp_path / "generated",
    )
    benchmark = run(
        "--action",
        "benchmark",
        "--model-path",
        trained["checkpoint"],
        "--input",
        path,
        "--synthetic-data",
        generated["synthetic_csv"],
        "--bootstrap-samples",
        2,
        "--output",
        tmp_path / "report",
    )
    assert benchmark["metrics"]["protocol"]["bootstrap_samples"] == 2


@pytest.mark.parametrize("model", ["vanilla-vae", "conv-vae", "timevae"])
def test_non_power_of_two_lengths_and_gradients(model):
    config = {
        "window_size": 9,
        "features": 1,
        "latent_dim": 2,
        "hidden_dims": [4, 8],
        "model_config": {},
    }
    network = TemporalModelRegistry().resolve(model).build(config)
    values = torch.randn(3, 9, 1)
    stats = network.objective(values, 1.0, torch.Generator().manual_seed(1))
    stats["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )
    assert network.sample(2, torch.Generator().manual_seed(1)).shape == (2, 9, 1)


def test_checkpoint_schema_validation_before_resolution(tmp_path):
    training = request(tmp_path)
    result = TrainTimeSeriesGenerator(training).execute()
    store, registry = TemporalCheckpointStore(), TemporalModelRegistry()
    checkpoint = store.load(result["checkpoint"], registry)
    del checkpoint["preprocessing"]["series_id_column"]
    store.save(tmp_path / "bad.pt", checkpoint)
    with pytest.raises(MLXUserError, match="Invalid temporal checkpoint"):
        store.load(tmp_path / "bad.pt", registry)


def test_invalid_generator_output_is_user_error_and_cleans_up(tmp_path):
    training = request(tmp_path)
    result = TrainTimeSeriesGenerator(training).execute()

    class BadGenerator:
        def sample(self, *args):
            yield np.zeros((2, 8, 3))

    with pytest.raises(MLXUserError, match="invalid shapes"):
        SynthesizeTimeSeries(
            SynthesizeRequest(
                model_path=result["checkpoint"],
                output_path=str(tmp_path / "bad"),
                num_sequences=2,
            ),
            backend=BadGenerator(),
        ).execute()
    assert not (tmp_path / "bad").exists()


def test_bootstrap_preserves_group_clusters_and_evaluator_reproducible():
    from mlx.modes.time_series.metrics import (
        TemporalQualityEvaluator,
        bootstrap_indices,
    )

    groups = np.array(["a"] * 4 + ["b"] * 4)
    indices = bootstrap_indices(groups, np.random.default_rng(1))
    assert all(np.all(np.diff(indices[i : i + 4]) == 1) for i in (0, 4))
    rng = np.random.default_rng(5)
    values = rng.normal(size=(8, 8, 1))
    metadata = {"feature_columns": ["value"], "mean": [0.0], "scale": [1.0]}
    request = BenchmarkRequest(bootstrap_samples=2, max_metric_windows=4)
    first = TemporalQualityEvaluator().evaluate(
        values, values, values, values, groups, metadata, request
    )
    second = TemporalQualityEvaluator().evaluate(
        values, values, values, values, groups, metadata, request
    )
    assert first == second
    assert first["protocol"]["evaluated_window_counts"]["real"] == 4
    assert (
        first["protocol"]["real_selected_span"][1]
        - first["protocol"]["real_selected_span"][0]
        == 4
    )
