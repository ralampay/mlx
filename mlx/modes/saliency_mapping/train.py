from __future__ import annotations

import platform
import time
import gc
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.saliency_mapping.checkpoints import (
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    training_paths,
)
from mlx.modes.saliency_mapping.data import SaliencyDataset, load_saliency_datasets
from mlx.modes.saliency_mapping.losses import SaliencyHybridLoss
from mlx.modes.saliency_mapping.metrics import SaliencyMetricAccumulator
from mlx.modes.saliency_mapping.models import build_saliency_model, grouped_model_names
from mlx.modes.saliency_mapping.requests import SaliencyRequest, TrainSaliencyRequest
from mlx.modes.saliency_mapping.samples import GenerateSaliencySamples


class TrainSaliencyModel:
    def __init__(
        self,
        request: TrainSaliencyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        model_registry=None,
    ) -> None:
        self.model_registry = model_registry
        self.request = request
        self.config = request.to_config()
        self.model_name = str(request.model)
        self.paths = training_paths(self.config, model_name=self.model_name)
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> dict[str, Any]:
        if self.request.epochs < 1:
            raise MLXUserError("--epochs must be at least 1 for saliency training.")
        train_dataset, validation_dataset = load_saliency_datasets(
            self.request.dataset_path,
            input_size=self.request.input_size,
            colored=self.request.colored,
            transform=self.request.transform,
        )
        workers = max(0, self.request.workers)
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, self.request.batch_size),
            shuffle=True,
            num_workers=workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=max(1, self.request.batch_size),
            shuffle=False,
            num_workers=workers,
        )
        model = build_saliency_model(self.model_name, self.config, **({"registry": self.model_registry} if self.model_registry is not None else {})).to(self.request.device)
        optimizer = Adam(model.parameters(), lr=float(self.request.lr or 1e-3))
        criterion = SaliencyHybridLoss(
            bce_weight=self.request.bce_weight,
            ssim_weight=self.request.ssim_weight,
            iou_weight=self.request.iou_weight,
        )
        start_epoch, best_mae, history = self._prepare_state(model, optimizer)
        if start_epoch >= self.request.epochs:
            raise MLXUserError(
                f"Resume checkpoint completed {start_epoch} epochs; --epochs must be greater."
            )
        self.paths["output_dir"].mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths["training_config_path"],
            {
                "config": self.config,
                "model_name": self.model_name,
                "checkpoint_selection": "lowest validation MAE",
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
            },
        )
        for epoch in range(start_epoch, self.request.epochs):
            epoch_start = time.perf_counter()
            training = self._run_epoch(model, train_loader, criterion, optimizer=optimizer)
            validation = self._run_epoch(model, validation_loader, criterion)
            row = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                **{f"train_{key}": value for key, value in training.items()},
                **{f"val_{key}": value for key, value in validation.items()},
            }
            history.append(row)
            write_csv(self.paths["training_csv_path"], history)
            self._write_training_curves(history)
            is_best = validation["mae"] < best_mae
            if is_best:
                best_mae = validation["mae"]
                save_checkpoint(
                    self.paths["checkpoint_path"],
                    model,
                    model_name=self.model_name,
                    config=self.config,
                )
            save_training_checkpoint(
                self.paths["last_checkpoint_path"],
                model,
                optimizer,
                model_name=self.model_name,
                config=self.config,
                completed_epoch=epoch + 1,
                best_validation_mae=best_mae,
                history=history,
            )
            emit(
                self.reporter,
                "progress",
                f"Completed saliency epoch {epoch + 1}/{self.request.epochs}.",
                current=epoch + 1,
                total=self.request.epochs,
                payload={
                    "event": "saliency_epoch",
                    "metrics": row,
                    "previous_metrics": history[-2] if len(history) > 1 else None,
                    "is_best_validation_mae": is_best,
                    "checkpoint_path": str(self.paths["last_checkpoint_path"]),
                    "best_checkpoint_path": str(self.paths["checkpoint_path"]),
                },
            )
        self._generate_optional_samples()
        result = {
            "model_name": self.model_name,
            "checkpoint_path": self.paths["checkpoint_path"],
            "last_checkpoint_path": self.paths["last_checkpoint_path"],
            "best_validation_mae": best_mae,
            "epochs": self.request.epochs,
        }
        emit(self.reporter, "success", "Saliency training complete.", payload=result)
        return result

    def _run_epoch(self, model, loader, criterion, optimizer=None) -> dict[str, float]:
        training = optimizer is not None
        model.train(training)
        totals = {name: 0.0 for name in ("loss", "bce_loss", "ssim_loss", "iou_loss")}
        accumulator = SaliencyMetricAccumulator(self.request.threshold_steps)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, targets in loader:
                images = images.to(self.request.device)
                targets = targets.to(self.request.device)
                if training:
                    optimizer.zero_grad()
                logits = model(images)
                losses = criterion(logits, targets).as_dict()
                if training:
                    losses["loss"].backward()
                    optimizer.step()
                for key, value in losses.items():
                    totals[key] += float(value.item()) * len(images)
                accumulator.update(
                    torch.sigmoid(logits).detach().cpu().numpy(),
                    targets.detach().cpu().numpy(),
                )
        metrics, _ = accumulator.finalize()
        return {
            **{key: value / max(1, len(loader.dataset)) for key, value in totals.items()},
            **{key: metrics[key] for key in ("mae", "max_f_beta", "mean_f_beta", "best_threshold", "precision", "recall")},
        }

    def _prepare_state(self, model, optimizer):
        if not self.request.model_path:
            write_csv(self.paths["training_csv_path"], [])
            return 0, float("inf"), []
        state = load_training_checkpoint(
            self.request.model_path,
            model,
            optimizer,
            model_name=self.model_name,
            config=self.config,
        )
        history = state["history"]
        write_csv(self.paths["training_csv_path"], history)
        if not self.paths["checkpoint_path"].is_file():
            save_checkpoint(
                self.paths["checkpoint_path"],
                model,
                model_name=self.model_name,
                config=self.config,
            )
        return state["completed_epoch"], state["best_validation_mae"], history

    def _generate_optional_samples(self) -> None:
        test_path = Path(self.request.dataset_path).expanduser() / "test"
        if not test_path.exists():
            return
        # Constructing the dataset validates incomplete and mismatched test partitions.
        SaliencyDataset(
            test_path,
            split="test",
            input_size=self.request.input_size,
            colored=self.request.colored,
            transform=self.request.transform,
        )
        GenerateSaliencySamples(
            self.config,
            checkpoint_path=self.paths["checkpoint_path"],
            split_path=test_path,
            output_dir=self.paths["output_dir"],
            reporter=self.reporter,
            **({"model_registry": self.model_registry} if self.model_registry is not None else {}),
        ).execute()

    def _write_training_curves(self, history):
        if not history:
            return
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
        axes[0].set(title="Saliency Hybrid Loss", ylabel="Loss")
        axes[0].legend()
        for name in ("val_mae", "val_max_f_beta", "val_mean_f_beta"):
            axes[1].plot(epochs, [row[name] for row in history], label=name)
        axes[1].set(title="Saliency Validation Metrics", xlabel="Epoch", ylabel="Score")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.paths["training_curves_path"], dpi=200)
        plt.close(figure)


class SmokeTestSaliencyModels:
    def __init__(
        self,
        request: SaliencyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        model_registry=None,
    ) -> None:
        self.model_registry = model_registry
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> list[dict[str, Any]]:
        results = []
        channels = 3 if self.request.colored else 1
        width, height = self.request.input_size
        for model_name in grouped_model_names(str(self.request.model), **({"registry": self.model_registry} if self.model_registry is not None else {})):
            model = build_saliency_model(model_name, self.request.to_config(), **({"registry": self.model_registry} if self.model_registry is not None else {})).to(self.request.device).eval()
            with torch.inference_mode():
                logits = model(
                    torch.randn(
                        max(1, self.request.batch_size), channels, height, width,
                        device=self.request.device,
                    )
                )
            expected = (max(1, self.request.batch_size), 1, height, width)
            if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
                raise MLXUserError(
                    f"Saliency model '{model_name}' produced invalid logits shape={tuple(logits.shape)}."
                )
            result = {
                "model_name": model_name,
                "input_shape": [expected[0], channels, height, width],
                "output_shape": list(logits.shape),
                "finite_logits": True,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
            results.append(result)
            emit(
                self.reporter,
                "success",
                f"Saliency smoke test passed for {model_name}.",
                payload={"event": "saliency_tensor_output", **result},
            )
            del model, logits
            gc.collect()
            if self.request.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


__all__ = ["SmokeTestSaliencyModels", "TrainSaliencyModel"]
