from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from mlx.core.artifacts import atomic_torch_save, write_csv, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.partitions import partition_rows, partition_hash
from mlx.modes.autoencoder.objectives import build_objective
from mlx.core.random import seed_everything
from mlx.modes.autoencoder.contracts import validate_model, validate_loss
from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
from mlx.modes.autoencoder.artifacts import (
    checkpoint_payload,
    prepare_output_directory,
    run_metadata,
    utc_timestamp,
    write_training_artifacts,
    write_transformed_csv,
)
from mlx.modes.autoencoder.data import (
    EmbeddingCsvLoader,
    MergeSingletonBatchSampler,
    VectorDataset,
    l2_normalize_tensor,
    load_json_object,
)
from mlx.modes.autoencoder.losses import (
    DEFAULT_LOSS_REGISTRY,
    ReconstructionLossRegistry,
)
from mlx.modes.autoencoder.models import (
    DEFAULT_AUTOENCODER_REGISTRY,
    AutoencoderRegistry,
)
from mlx.modes.autoencoder.requests import AutoencoderEmbedRequest, AutoencoderTrainRequest


@dataclass(frozen=True)
class AutoencoderTrainingResult:
    output_dir: Path
    checkpoint_path: Path
    input_dimensions: int
    bottleneck_dimensions: int
    best_validation_loss: float


@dataclass(frozen=True)
class AutoencoderEmbeddingResult:
    output_path: Path
    rows: int
    input_dimensions: int
    output_dimensions: int


class TrainAutoencoder:
    def __init__(
        self,
        request: AutoencoderTrainRequest,
        *,
        reporter: WorkflowReporter | None = None,
        model_registry: AutoencoderRegistry = DEFAULT_AUTOENCODER_REGISTRY,
        loss_registry: ReconstructionLossRegistry = DEFAULT_LOSS_REGISTRY,
        csv_loader: EmbeddingCsvLoader | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.model_registry = model_registry
        self.loss_registry = loss_registry
        self.csv_loader = csv_loader or EmbeddingCsvLoader()

    def execute(self) -> AutoencoderTrainingResult:
        self._validate_request()
        if type(self.request.minimum_batch_size) is not int or self.request.minimum_batch_size not in (1, 2):
            raise MLXUserError("Autoencoder minimum_batch_size must be 1 or 2.")
        seed_everything(self.request.random_seed)
        table = self.csv_loader.load(str(self.request.input_path))
        if self.request.input_dim is not None and self.request.input_dim != table.dimensions:
            raise MLXUserError(
                f"--input-dim is {self.request.input_dim}, but the CSV contains {table.dimensions}-dimensional vectors."
            )
        normalize_inputs = self._effective_normalization(table.source_normalized)
        model_options = load_json_object(
            self.request.autoencoder_config, purpose="autoencoder"
        )
        reserved = {"input_dimensions", "hidden_dimensions", "bottleneck_dimensions"}
        overlap = sorted(reserved & model_options.keys())
        if overlap:
            raise MLXUserError(
                "Autoencoder configuration cannot override dimension flags: "
                + ", ".join(overlap)
            )
        model_config = {
            **model_options,
            "input_dimensions": table.dimensions,
            "hidden_dimensions": self.request.hidden_dim,
            "bottleneck_dimensions": self.request.bottleneck_dim,
        }
        definition, architecture_path = self.model_registry.resolve(str(self.request.model))
        try:
            model = definition.build(model_config)
        except MLXUserError:
            raise
        except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise MLXUserError(
                f"Unable to build autoencoder model '{definition.name}': {exc}"
            ) from exc
        self._validate_model(model, table.dimensions)
        loss_options = load_json_object(self.request.loss_config, purpose="loss")
        loss_definition, loss_path = self.loss_registry.resolve(self.request.loss)
        try:
            loss_options = {**getattr(loss_definition, "default_config", {}), **loss_options}
            criterion = loss_definition.build(loss_options)
        except MLXUserError:
            raise
        except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise MLXUserError(
                f"Unable to build autoencoder loss '{loss_definition.name}': {exc}"
            ) from exc
        if not isinstance(criterion, torch.nn.Module):
            raise MLXUserError("Autoencoder loss definitions must build a torch.nn.Module.")
        try:
            model.to(self.request.device)
            criterion.to(self.request.device)
        except (RuntimeError, ValueError) as exc:
            raise MLXUserError(f"Unable to initialize autoencoder on device '{self.request.device}': {exc}") from exc
        minimum_batch_size = getattr(criterion, "minimum_batch_size", 1)
        if type(minimum_batch_size) is not int or minimum_batch_size not in (1, 2):
            raise MLXUserError("Loss minimum_batch_size must be 1 or 2.")
        minimum_batch_size = max(self.request.minimum_batch_size, minimum_batch_size)
        train_loader, validation_loader = self._loaders(
            table.vectors, normalize_inputs, minimum_batch_size=minimum_batch_size,
        )
        self._objective = build_objective(model, criterion, int(self.request.random_seed or 0))
        self._epoch_components = {}
        training_indices = train_loader.dataset.indices
        validation_indices = validation_loader.dataset.indices
        split_hash = partition_hash(training_indices, validation_indices)
        initializer = getattr(model, "initialize_from_training_values", None)
        if initializer is not None:
            initializer(train_loader.dataset.dataset.values[sorted(training_indices)].to(self.request.device))
        if hasattr(criterion, "calibrate"):
            calibration_indices = sorted(training_indices)[:1024]
            values = train_loader.dataset.dataset.values[calibration_indices].to(self.request.device)
            criterion.calibrate(model, values)
        output_dir = prepare_output_directory(str(self.request.output_path))
        started_at = utc_timestamp()
        emit(
            self.reporter,
            "info",
            f"Training {definition.name} on {len(table.vectors)} vectors.",
            payload={"event": "autoencoder_training_started", "rows": len(table.vectors)},
        )
        try:
            optimizer = Adam(model.parameters(), lr=float(self.request.lr or 0.001))
        except (TypeError, ValueError) as exc:
            raise MLXUserError(f"Unable to construct the autoencoder optimizer: {exc}") from exc
        history: list[dict[str, float | int]] = []
        best_loss = float("inf")
        best_epoch = 0
        best_checkpoint: dict[str, Any] | None = None
        for epoch in range(0 if initializer is not None else 1, self.request.epochs + 1):
            train_loss = (self._validation_loss(model, train_loader, criterion) if epoch == 0 else
                          self._train_epoch(model, train_loader, criterion, optimizer))
            train_components = dict(self._epoch_components)
            validation_loss = self._validation_loss(model, validation_loader, criterion)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{f"train_{k}": v for k, v in train_components.items()},
                **{f"val_{k}": v for k, v in self._epoch_components.items()},
            }
            history.append(row)
            write_csv(output_dir / "training.csv", history)
            improved = validation_loss < best_loss
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
            if improved or not self.request.use_best:
                best_checkpoint = checkpoint_payload(
                    model=model,
                    architecture_name=definition.name,
                    architecture_path=architecture_path,
                    model_config=model_config,
                    loss_name=loss_definition.name,
                    loss_path=loss_path,
                    loss_config=loss_options,
                    expects_l2_normalized_input=normalize_inputs,
                    best_epoch=best_epoch if self.request.use_best else epoch,
                    best_validation_loss=best_loss if self.request.use_best else validation_loss,
                    source_path=table.path,
                )
                best_checkpoint.update(
                    initialization=getattr(model, "initialization_metadata", {}),
                    calibration=getattr(criterion, "calibration", {}),
                    split_hash=split_hash, seed=int(self.request.random_seed or 0),
                    sampling_seed=int(self.request.random_seed or 0),
                    validation_components=dict(self._epoch_components),
                )
                atomic_torch_save(best_checkpoint, output_dir / "autoencoder.pth")
            emit(
                self.reporter,
                "progress",
                f"Epoch {epoch}/{self.request.epochs}: train={train_loss:.6f}, validation={validation_loss:.6f}",
                current=epoch,
                total=self.request.epochs,
                payload={"event": "autoencoder_training_epoch", "metrics": row},
            )
        if best_checkpoint is None:
            raise MLXUserError("Autoencoder training did not produce a checkpoint.")
        metadata = run_metadata(
            action="train",
            started_at=started_at,
            values={
                "architecture": definition.name,
                "loss": loss_definition.name,
                "rows": len(table.vectors),
                "input_dimensions": table.dimensions,
                "bottleneck_dimensions": self.request.bottleneck_dim,
                "expects_l2_normalized_input": normalize_inputs,
                "best_epoch": best_checkpoint["best_epoch"],
                "best_validation_loss": best_checkpoint["best_validation_loss"],
                "seed": self.request.random_seed,
                "minimum_batch_size": minimum_batch_size,
                "device": self.request.device,
                "split_hash": split_hash,
                "initialization": getattr(model, "initialization_metadata", {}),
                "calibration": getattr(criterion, "calibration", {}),
                "validation_components": best_checkpoint["validation_components"],
            },
        )
        write_training_artifacts(
            output_dir,
            checkpoint=best_checkpoint,
            history=history,
            metadata=metadata,
            plots=self.request.plots,
        )
        result = AutoencoderTrainingResult(
            output_dir,
            output_dir / "autoencoder.pth",
            table.dimensions,
            self.request.bottleneck_dim,
            float(best_checkpoint["best_validation_loss"]),
        )
        emit(
            self.reporter,
            "success",
            f"Autoencoder training complete: {result.checkpoint_path}",
            payload={"event": "autoencoder_training_completed", "result": result},
        )
        return result

    def _validate_request(self) -> None:
        if not self.request.model:
            raise MLXUserError("Autoencoder training requires --model.")
        if not self.request.input_path:
            raise MLXUserError("Autoencoder training requires --input.")
        if not self.request.output_path:
            raise MLXUserError("Autoencoder training requires --output.")
        if self.request.input_dim is not None and self.request.input_dim < 2:
            raise MLXUserError("--input-dim must be at least 2.")
        if self.request.bottleneck_dim < 1:
            raise MLXUserError("--bottleneck-dim must be positive.")
        if self.request.epochs < 1 or self.request.batch_size < 1:
            raise MLXUserError("--epochs and --batch-size must be positive.")
        if self.request.workers < 0:
            raise MLXUserError("--workers cannot be negative.")
        if self.request.lr is not None and self.request.lr <= 0:
            raise MLXUserError("--lr must be positive.")
        ratio = float(self.request.val_ratio or 0.0)
        if not 0.0 < ratio < 1.0:
            raise MLXUserError("--val-ratio must be strictly between zero and one.")

    def _effective_normalization(self, source_normalized: bool) -> bool:
        if source_normalized and self.request.normalize_inputs is False:
            raise MLXUserError(
                "--no-normalize-inputs conflicts with the sibling embedding manifest, "
                "which records already-normalized vectors."
            )
        return source_normalized or self.request.normalize_inputs is True

    def _validate_model(self, model, input_dimensions: int) -> None:
        validate_model(model, input_dimensions=input_dimensions,
                       bottleneck_dimensions=self.request.bottleneck_dim)

    def _loaders(self, vectors, normalize_inputs: bool, *, minimum_batch_size: int = 1):
        if len(vectors) < 2:
            raise MLXUserError("Autoencoder training requires at least two vectors.")
        dataset = VectorDataset(vectors)
        if normalize_inputs:
            dataset.values = l2_normalize_tensor(dataset.values)
        generator = torch.Generator().manual_seed(int(self.request.random_seed or 0))
        train_indices, val_indices = partition_rows(
            len(dataset), float(self.request.val_ratio), int(self.request.random_seed or 0), generator=generator,
        )
        validation = Subset(dataset, val_indices)
        training = Subset(dataset, train_indices)
        if type(minimum_batch_size) is not int or minimum_batch_size not in (1, 2):
            raise MLXUserError("Autoencoder losses may request minimum_batch_size 1 or 2.")
        if minimum_batch_size == 2:
            if self.request.batch_size < 2:
                raise MLXUserError("This autoencoder loss requires --batch-size of at least 2.")
            if min(len(training), len(validation)) < 2:
                raise MLXUserError(
                    "This autoencoder loss requires at least two vectors in each training and "
                    "validation partition. Provide more vectors or adjust --val-ratio."
                )
            return (
                DataLoader(
                    training,
                    batch_sampler=MergeSingletonBatchSampler(
                        RandomSampler(training, generator=generator), self.request.batch_size,
                    ),
                    num_workers=self.request.workers,
                    generator=generator,
                ),
                DataLoader(
                    validation,
                    batch_sampler=MergeSingletonBatchSampler(
                        SequentialSampler(validation), self.request.batch_size,
                    ),
                    num_workers=self.request.workers,
                ),
            )
        return (
            DataLoader(
                training,
                batch_size=self.request.batch_size,
                shuffle=True,
                num_workers=self.request.workers,
                generator=generator,
            ),
            DataLoader(
                validation,
                batch_size=self.request.batch_size,
                shuffle=False,
                num_workers=self.request.workers,
            ),
        )

    def _train_epoch(self, model, loader, criterion, optimizer) -> float:
        model.train()
        total = 0.0
        count = 0
        components = {}
        for values in loader:
            values = values.to(self.request.device)
            optimizer.zero_grad()
            try:
                evaluation = self._objective.evaluate(model, values, training=model.training)
                loss = evaluation.loss
                for key, value in evaluation.components.items():
                    components[key] = components.get(key, 0.0) + float(value.detach()) * len(values)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise MLXUserError(f"Autoencoder training step failed: {exc}") from exc
            validate_loss(loss, training=True)
            loss.backward()
            optimizer.step()
            projector = getattr(model, "project_parameters_", None)
            if projector is not None:
                try:
                    projector()
                except RuntimeError as exc:
                    raise MLXUserError(f"Unable to enforce model parameter constraints: {exc}") from exc
            total += float(loss.item()) * values.shape[0]
            count += values.shape[0]
        if not count:
            raise MLXUserError("Autoencoder training partition is empty.")
        self._epoch_components = {key: value / count for key, value in components.items()}
        diagnostics = getattr(model, "training_diagnostics", None)
        if diagnostics is not None:
            self._epoch_components.update({key: float(value) for key, value in diagnostics().items()})
        return total / count

    @torch.no_grad()
    def _validation_loss(self, model, loader, criterion) -> float:
        model.eval()
        total = 0.0
        count = 0
        components = {}
        for values in loader:
            values = values.to(self.request.device)
            try:
                evaluation = self._objective.evaluate(model, values, training=model.training)
                loss = evaluation.loss
                for key, value in evaluation.components.items():
                    components[key] = components.get(key, 0.0) + float(value.detach()) * len(values)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise MLXUserError(f"Autoencoder validation step failed: {exc}") from exc
            validate_loss(loss, training=False)
            total += float(loss.item()) * values.shape[0]
            count += values.shape[0]
        if not count:
            raise MLXUserError("Autoencoder validation partition is empty.")
        self._epoch_components = {key: value / count for key, value in components.items()}
        diagnostics = getattr(model, "training_diagnostics", None)
        if diagnostics is not None:
            self._epoch_components.update({key: float(value) for key, value in diagnostics().items()})
        return total / count


class EmbedAutoencoder:
    def __init__(
        self,
        request: AutoencoderEmbedRequest,
        *,
        reporter: WorkflowReporter | None = None,
        csv_loader: EmbeddingCsvLoader | None = None,
        transformer_factory=AutoencoderRepresentationTransformer,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.csv_loader = csv_loader or EmbeddingCsvLoader()
        self.transformer_factory = transformer_factory

    def execute(self) -> AutoencoderEmbeddingResult:
        if not self.request.model_path:
            raise MLXUserError("Autoencoder embed requires --model-path.")
        if not self.request.input_path or not self.request.output_path:
            raise MLXUserError("Autoencoder embed requires --input and --output.")
        if self.request.batch_size < 1:
            raise MLXUserError("--batch-size must be positive.")
        table = self.csv_loader.load(self.request.input_path)
        transformer = self.transformer_factory(
            self.request.model_path, device=self.request.device,
            **({"trust_checkpoint_code": True} if self.request.trust_checkpoint_code else {}),
            **({"output_dimensions": self.request.output_dim} if self.request.output_dim is not None else {}),
        )
        if table.dimensions != transformer.input_dimensions:
            raise MLXUserError(
                f"Autoencoder expects {transformer.input_dimensions} dimensions, "
                f"but the CSV contains {table.dimensions}."
            )
        provenance = transformer.provenance
        expected_model = provenance.get("architecture")
        allowed_models = {expected_model, provenance.get("architecture_path")}
        if self.request.model and self.request.model not in allowed_models:
            raise MLXUserError(
                f"--model '{self.request.model}' does not match checkpoint architecture '{expected_model}'."
            )
        started_at = utc_timestamp()
        output_path = Path(self.request.output_path).expanduser()
        manifest_path = output_path.with_suffix(".manifest.json")
        if output_path.exists() or manifest_path.exists():
            existing = output_path if output_path.exists() else manifest_path
            raise MLXUserError(f"Autoencoder embedding output already exists: {existing}")
        transformed: list[list[float]] = []
        for start in range(0, len(table.vectors), self.request.batch_size):
            inputs = table.vectors[start : start + self.request.batch_size]
            batch = transformer.transform(inputs)
            if len(batch) != len(inputs):
                raise MLXUserError(
                    f"Autoencoder returned {len(batch)} vectors for {len(inputs)} inputs."
                )
            if any(len(vector) != transformer.output_dimensions for vector in batch):
                raise MLXUserError(
                    "Autoencoder output does not match its declared bottleneck dimensions."
                )
            if any(not math.isfinite(float(value)) for vector in batch for value in vector):
                raise MLXUserError("Autoencoder produced non-finite bottleneck values.")
            if self.request.normalize_embeddings:
                batch = [self._normalize(vector) for vector in batch]
            transformed.extend(batch)
            emit(
                self.reporter,
                "progress",
                f"Encoded {len(transformed)} of {len(table.vectors)} vectors.",
                current=len(transformed),
                total=len(table.vectors),
                payload={"event": "autoencoder_embedding_progress"},
            )
        write_transformed_csv(output_path, table, transformed)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "input": table.path.name,
                "output": output_path.name,
                "rows": len(table.vectors),
                "input_dimensions": transformer.input_dimensions,
                "output_dimensions": transformer.output_dimensions,
                "normalized": self.request.normalize_embeddings,
                "adapter": dict(transformer.provenance),
                "run": run_metadata(action="embed", started_at=started_at, values={}),
            },
        )
        result = AutoencoderEmbeddingResult(
            output_path,
            len(table.vectors),
            transformer.input_dimensions,
            transformer.output_dimensions,
        )
        emit(
            self.reporter,
            "success",
            f"Autoencoder embeddings written to {output_path}.",
            payload={"event": "autoencoder_embedding_completed", "result": result},
        )
        return result

    @staticmethod
    def _normalize(vector) -> list[float]:
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if norm == 0.0:
            raise MLXUserError("Cannot L2-normalize a zero bottleneck vector.")
        return [float(value) / norm for value in vector]


class ListAutoencoderModels:
    def __init__(self, registry: AutoencoderRegistry = DEFAULT_AUTOENCODER_REGISTRY) -> None:
        self.registry = registry

    def execute(self) -> tuple[Mapping[str, str], ...]:
        return tuple(
            {
                "name": name,
                "description": self.registry.descriptions.get(name, ""),
            }
            for name in self.registry.names()
        )


class ListAutoencoderLosses:
    def __init__(self, registry: ReconstructionLossRegistry = DEFAULT_LOSS_REGISTRY) -> None:
        self.registry = registry

    def execute(self) -> tuple[Mapping[str, str], ...]:
        return tuple(
            {
                "name": name,
                "description": self.registry.descriptions.get(name, ""),
            }
            for name in self.registry.names()
        )


__all__ = [
    "AutoencoderEmbeddingResult",
    "AutoencoderTrainingResult",
    "EmbedAutoencoder",
    "ListAutoencoderLosses",
    "ListAutoencoderModels",
    "TrainAutoencoder",
]
