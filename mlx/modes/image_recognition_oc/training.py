from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from mlx.core.artifacts import write_csv
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.random import restore_random_state, seed_everything
from mlx.modes.image_recognition_oc.algorithms import (
    DEFAULT_ALGORITHM_REGISTRY,
    OneClassAlgorithmRegistry,
)
from mlx.modes.image_recognition_oc.artifacts import (
    load_raw_checkpoint,
    resolve_training_paths,
    save_deployment_checkpoint,
    save_training_checkpoint,
    update_checkpoint_model,
    utc_timestamp,
    validate_resume_checkpoint,
    write_json,
    write_training_plot,
)
from mlx.modes.image_recognition_oc.data import OneClassImageDataset
from mlx.modes.image_recognition_oc.requests import TrainImageOneClassRequest


TRAINING_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_score_mean",
    "train_score_std",
    "val_score_mean",
    "val_score_std",
    "val_score_p50",
    "val_score_p95",
    "learning_rate",
)


@dataclass(frozen=True)
class ImageOneClassTrainingData:
    train_dataset: Any
    validation_dataset: Any
    train_loader: DataLoader
    center_loader: DataLoader
    validation_loader: DataLoader


@torch.no_grad()
def collect_scores(algorithm, model, loader, device: str) -> torch.Tensor:
    model.eval()
    batches = [algorithm.scores(model, images.to(device)).detach().cpu() for images, _, _ in loader]
    if not batches:
        raise MLXUserError("Cannot score an empty one-class image dataset.")
    scores = torch.cat(batches)
    if not torch.all(torch.isfinite(scores)):
        raise MLXUserError("One-class image scoring produced non-finite values.")
    return scores


class TrainImageOneClassModel:
    def __init__(
        self,
        request: TrainImageOneClassRequest,
        *,
        reporter: WorkflowReporter | None = None,
        registry: OneClassAlgorithmRegistry = DEFAULT_ALGORITHM_REGISTRY,
        dataset_factory=OneClassImageDataset,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.registry = registry
        self.dataset_factory = dataset_factory

    def execute(self) -> dict[str, Any]:
        config = self.request.to_config()
        self._validate(config)
        seed_everything(config.get("random_seed"))
        algorithm = self.registry.get(str(config["model"]))
        paths = resolve_training_paths(config)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        data = self._training_data(config)
        device = str(config["device"])
        model_config = {**config, "pretrained": False} if config.get("model_path") else config
        model = algorithm.build_model(str(config["backbone"]), model_config).to(device)
        optimizer = Adam(model.parameters(), lr=float(config.get("lr") or 0.001))
        start_epoch, best, history, resumed = self._prepare_resume(
            model, optimizer, config, paths["training_csv"]
        )
        if not resumed:
            emit(self.reporter, "info", "Initializing Deep SVDD center from normal training images")
            algorithm.initialize(model, data.center_loader, device)

        epochs = int(config["epochs"])
        for epoch in range(start_epoch, epochs):
            train_scores, train_loss = self._train_epoch(
                algorithm, model, data.train_loader, optimizer, device
            )
            val_scores = collect_scores(algorithm, model, data.validation_loader, device)
            val_loss = float(val_scores.mean().item())
            row = self._history_row(
                epoch + 1,
                train_loss,
                train_scores,
                val_scores,
                float(optimizer.param_groups[0]["lr"]),
            )
            history.append(row)
            write_csv(paths["training_csv"], history, fieldnames=TRAINING_COLUMNS)
            improved = val_loss < best
            if improved:
                best = val_loss
            if improved or not bool(config.get("use_best", True)):
                save_deployment_checkpoint(paths["checkpoint"], model, config, algorithm)
            save_training_checkpoint(
                paths["last_checkpoint"],
                model,
                optimizer,
                config,
                algorithm,
                completed_epoch=epoch + 1,
                history=history,
                best_validation_objective=best,
            )
            emit(
                self.reporter,
                "progress",
                f"Epoch {epoch + 1}/{epochs}: train={train_loss:.6f}, validation={val_loss:.6f}",
                current=epoch + 1,
                total=epochs,
                payload={"event": "image_one_class_training_epoch", "metrics": row, "best": best},
            )

        if not paths["checkpoint"].is_file():
            save_deployment_checkpoint(paths["checkpoint"], model, config, algorithm)
        self._calibrate_checkpoint(paths["checkpoint"], data.validation_loader, config, device)
        self._calibrate_checkpoint(paths["last_checkpoint"], data.validation_loader, config, device)
        write_training_plot(paths["training_plot"], history)
        deployment = load_raw_checkpoint(paths["checkpoint"])
        completed_epoch = int(history[-1]["epoch"]) if history else start_epoch
        metadata = {
            "mode": "image_recognition_oc",
            "created_at": utc_timestamp(),
            "model": config["model"],
            "backbone": config["backbone"],
            "dataset": str(config["dataset_path"]),
            "normal_only_training": True,
            "train_images": len(data.train_dataset),
            "validation_normal_images": len(data.validation_dataset),
            "epochs_completed": completed_epoch,
            "seed": config.get("random_seed"),
            "device": device,
            "threshold": float(deployment["threshold"]),
            "threshold_quantile": float(config["svdd_quantile"]),
            "score_type": deployment["score_type"],
            "checkpoint": paths["checkpoint"].name,
            "last_checkpoint": paths["last_checkpoint"].name,
        }
        write_json(paths["run_metadata"], metadata)
        emit(
            self.reporter,
            "success",
            f"One-class image training complete; threshold={metadata['threshold']:.6f}",
            payload={"event": "image_one_class_training_complete", **metadata},
        )
        return {"paths": paths, "history": history, "metadata": metadata}

    def _training_data(self, config: dict[str, Any]) -> ImageOneClassTrainingData:
        common = {
            "height": int(config["height"]),
            "width": int(config["width"]),
            "colored": bool(config["colored"]),
            "normal_only": True,
        }
        train = self.dataset_factory(
            config["dataset_path"],
            split="train",
            augment=bool(config.get("apply_transformations", False)),
            **common,
        )
        center = self.dataset_factory(
            config["dataset_path"], split="train", augment=False, **common
        )
        validation = self.dataset_factory(
            config["dataset_path"], split="val", augment=False, **common
        )
        loader = lambda dataset, shuffle: DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle,
            num_workers=int(config.get("workers", 0)),
        )
        return ImageOneClassTrainingData(
            train_dataset=train,
            validation_dataset=validation,
            train_loader=loader(train, True),
            center_loader=loader(center, False),
            validation_loader=loader(validation, False),
        )

    @staticmethod
    def _train_epoch(algorithm, model, loader, optimizer, device: str):
        model.train()
        batches = []
        for images, labels, _ in loader:
            if torch.any(labels != 0):
                raise MLXUserError("An anomalous image reached the normal-only training loader.")
            optimizer.zero_grad()
            loss, batch_scores = algorithm.training_step(model, images.to(device))
            if not torch.isfinite(loss) or not torch.all(torch.isfinite(batch_scores)):
                raise MLXUserError("One-class image training produced a non-finite loss or score.")
            loss.backward()
            optimizer.step()
            batches.append(batch_scores.detach().cpu())
        if not batches:
            raise MLXUserError("Cannot train on an empty normal image dataset.")
        scores = torch.cat(batches)
        return scores, float(scores.mean().item())

    def _prepare_resume(self, model, optimizer, config, training_csv: Path):
        if not config.get("model_path"):
            return 0, float("inf"), [], False
        checkpoint = load_raw_checkpoint(config["model_path"])
        algorithm = self.registry.get(str(config["model"]))
        validate_resume_checkpoint(checkpoint, config, algorithm)
        config["pretrained_initialization"] = bool(
            checkpoint.get("pretrained_initialization", False)
        )
        try:
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            history = list(checkpoint["history"])
            completed = int(checkpoint["completed_epoch"])
            best = float(checkpoint["best_validation_objective"])
            restore_random_state(checkpoint["random_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise MLXUserError(f"Could not restore one-class training checkpoint: {exc}") from exc
        if int(config["epochs"]) <= completed:
            raise MLXUserError(
                f"--epochs must exceed the checkpoint's {completed} completed epochs when resuming."
            )
        write_csv(training_csv, history, fieldnames=TRAINING_COLUMNS)
        return completed, best, history, True

    def _calibrate_checkpoint(self, path: Path, validation_loader, config, device: str) -> None:
        checkpoint = load_raw_checkpoint(path)
        algorithm = self.registry.get(str(checkpoint["model_name"]))
        stored_config = {
            **config,
            "model": checkpoint["model_name"],
            "backbone": checkpoint["backbone_name"],
            "pretrained": False,
        }
        calibrated = algorithm.build_model(str(checkpoint["backbone_name"]), stored_config).to(device)
        try:
            calibrated.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as exc:
            raise MLXUserError(f"Could not reload checkpoint '{path}' for calibration: {exc}") from exc
        scores = collect_scores(algorithm, calibrated, validation_loader, device)
        algorithm.calibrate(calibrated, scores, stored_config)
        update_checkpoint_model(path, checkpoint, calibrated, stored_config, algorithm)

    @staticmethod
    def _history_row(epoch, train_loss, train_scores, val_scores, learning_rate):
        return {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_scores.mean().item()),
            "train_score_mean": float(train_scores.mean().item()),
            "train_score_std": float(train_scores.std(unbiased=False).item()),
            "val_score_mean": float(val_scores.mean().item()),
            "val_score_std": float(val_scores.std(unbiased=False).item()),
            "val_score_p50": float(torch.quantile(val_scores, 0.5).item()),
            "val_score_p95": float(torch.quantile(val_scores, 0.95).item()),
            "learning_rate": learning_rate,
        }

    def _validate(self, config: dict[str, Any]) -> None:
        if not config.get("model"):
            raise MLXUserError("Training requires --model.")
        if not config.get("backbone"):
            raise MLXUserError("Training requires --backbone.")
        self.registry.get(str(config["model"]))
        if int(config.get("epochs", 0)) < 1:
            raise MLXUserError("--epochs must be at least 1.")
        if int(config.get("batch_size", 0)) < 1:
            raise MLXUserError("--batch-size must be at least 1.")
        if int(config.get("workers", 0)) < 0:
            raise MLXUserError("--workers must be greater than or equal to zero.")
        if float(config.get("lr") or 0) <= 0:
            raise MLXUserError("--lr must be greater than zero.")


__all__ = ["TrainImageOneClassModel", "collect_scores"]
