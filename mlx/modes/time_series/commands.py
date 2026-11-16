"""Use-case orchestration; replaceable collaborators are constructor-injected."""

import numpy as np

from mlx.core.commands import NullWorkflowReporter
from mlx.core.exceptions import MLXUserError

from .artifacts import (
    FORMAT,
    SyntheticCsvWriter,
    TemporalCheckpointStore,
    output_directory,
    write_json,
)
from .data import TemporalCsvLoader, file_digest, load_synthetic
from .registry import TemporalModelRegistry
from .requests import positive_int


class TrainTimeSeriesGenerator:
    def __init__(
        self,
        request,
        *,
        loader=None,
        registry=None,
        backend=None,
        store=None,
        reporter=None,
    ):
        self.request = request
        self.loader = loader if loader is not None else TemporalCsvLoader()
        self.registry = registry if registry is not None else TemporalModelRegistry()
        self.backend = backend
        self.store = store if store is not None else TemporalCheckpointStore()
        self.reporter = reporter if reporter is not None else NullWorkflowReporter()

    def execute(self):
        request = self.request
        request.validate()
        definition = self.registry.resolve(request.model)
        prepared = self.loader.load(
            request.input_path,
            time_column=request.time_column,
            feature_columns=request.feature_columns,
            series_id_column=request.series_id_column,
            window_size=request.window_size,
            stride=request.stride,
        )
        training, _ = prepared.windows("train")
        validation, _ = prepared.windows("val", stride=request.window_size)
        config = {
            "window_size": request.window_size,
            "features": len(request.feature_columns),
            "latent_dim": request.latent_dim,
            "hidden_dims": list(request.hidden_dims),
            "model_config": {
                **getattr(definition, "default_config", {}),
                **(request.model_config or {}),
            },
        }
        if self.backend is None:
            from .backend import TorchTemporalBackend

            backend = TorchTemporalBackend()
        else:
            backend = self.backend
        with output_directory(request.output_path) as output:
            state, history = backend.fit(
                definition, config, training, validation, request, self.reporter
            )
            checkpoint = {
                "format": FORMAT,
                "model": request.model,
                "definition": self.registry.entries[request.model],
                "config": config,
                "preprocessing": prepared.metadata,
                "state_dict": state,
                "training": {
                    "seed": request.random_seed,
                    "epochs": request.epochs,
                    "batch_size": request.batch_size,
                    "lr": request.lr,
                    "kl_weight": request.kl_weight,
                    "objective": "mean reconstruction MSE + kl_weight * mean latent KL",
                },
            }
            self.store.save(output / "best.pt", checkpoint)
            write_json(output / "history.json", history)
            write_json(
                output / "manifest.json",
                {
                    key: value
                    for key, value in checkpoint.items()
                    if key != "state_dict"
                },
            )
        return {
            "output": str(request.output_path),
            "checkpoint": str(request.output_path) + "/best.pt",
            "model": request.model,
            "training_windows": len(training),
            "validation_windows": len(validation),
            "best_validation_loss": min(row["val"]["loss"] for row in history),
        }


class SynthesizeTimeSeries:
    def __init__(
        self, request, *, registry=None, backend=None, store=None, writer=None
    ):
        self.request = request
        self.registry = registry if registry is not None else TemporalModelRegistry()
        self.backend = backend
        self.store = store if store is not None else TemporalCheckpointStore()
        self.writer = writer if writer is not None else SyntheticCsvWriter()

    def execute(self):
        request = self.request
        for name in ("num_sequences", "batch_size"):
            positive_int(getattr(request, name), name)
        positive_int(request.random_seed, "seed", 0)
        checkpoint = self.store.load(request.model_path, self.registry)
        definition = self.registry.resolve(checkpoint["model"])
        metadata = checkpoint["preprocessing"]
        if self.backend is None:
            from .backend import TorchTemporalBackend

            backend = TorchTemporalBackend()
        else:
            backend = self.backend
        with output_directory(request.output_path) as output:
            count = self.writer.write(
                output / "synthetic.csv",
                backend.sample(definition, checkpoint, request),
                metadata,
                request.num_sequences,
            )
            write_json(
                output / "manifest.json",
                {
                    "model": checkpoint["model"],
                    "checkpoint_sha256": file_digest(request.model_path),
                    "csv_sha256": file_digest(output / "synthetic.csv"),
                    "seed": request.random_seed,
                    "batch_size": request.batch_size,
                    "num_sequences": count,
                    "preprocessing": metadata,
                    "generation": "independent prior samples",
                },
            )
        return {
            "output": str(request.output_path),
            "synthetic_csv": str(request.output_path) + "/synthetic.csv",
            "num_sequences": count,
        }


class BenchmarkSyntheticTimeSeries:
    def __init__(
        self,
        request,
        *,
        loader=None,
        registry=None,
        store=None,
        evaluator=None,
        report_writer=None,
    ):
        self.request = request
        self.loader = loader if loader is not None else TemporalCsvLoader()
        self.registry = registry if registry is not None else TemporalModelRegistry()
        self.store = store if store is not None else TemporalCheckpointStore()
        self.evaluator = evaluator
        self.report_writer = report_writer

    def execute(self):
        request = self.request
        positive_int(request.random_seed, "seed", 0)
        positive_int(request.bootstrap_samples, "bootstrap-samples")
        positive_int(request.max_metric_windows, "max-metric-windows", 2)
        checkpoint = self.store.load(request.model_path, self.registry)
        metadata = checkpoint["preprocessing"]
        prepared = self.loader.load(
            request.input_path,
            **{
                key: metadata[key]
                for key in (
                    "time_column",
                    "feature_columns",
                    "series_id_column",
                    "window_size",
                    "stride",
                )
            },
            expected=metadata,
        )
        arrays = {
            split: prepared.windows(split, stride=metadata["window_size"])
            for split in ("train", "val", "test")
        }
        training_reference, _ = prepared.windows("train")
        synthetic = load_synthetic(request.synthetic_data, metadata)
        synthetic = (synthetic - metadata["mean"]) / metadata["scale"]
        if not np.isfinite(synthetic).all():
            raise MLXUserError(
                "Synthetic data exceeds supported standardized numeric range."
            )
        if self.evaluator is None:
            from .metrics import TemporalQualityEvaluator

            evaluator = TemporalQualityEvaluator()
        else:
            evaluator = self.evaluator
        if self.report_writer is None:
            from .presentation import TemporalReportWriter

            report_writer = TemporalReportWriter()
        else:
            report_writer = self.report_writer
        with output_directory(request.output_path) as output:
            metrics = evaluator.evaluate(
                arrays["train"][0],
                arrays["val"][0],
                arrays["test"][0],
                synthetic,
                arrays["test"][1],
                metadata,
                request,
                training_reference=training_reference,
            )
            metrics["provenance"] = {
                "checkpoint_sha256": file_digest(request.model_path),
                "reference_sha256": metadata["source_sha256"],
                "synthetic_sha256": file_digest(request.synthetic_data),
            }
            write_json(output / "metrics.json", metrics)
            report_writer.write(output, metrics, arrays["test"][0], synthetic, metadata)
        return {
            "output": str(request.output_path),
            "report": str(request.output_path) + "/report.html",
            "metrics": metrics,
        }
