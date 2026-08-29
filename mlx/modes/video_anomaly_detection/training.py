from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.deep_svdd import quantile_threshold
from mlx.core.exceptions import MLXUserError
from mlx.core.random import seed_everything
from mlx.modes.video_anomaly_detection.artifacts import (
    resolve_training_paths,
    restore_random_state,
    save_deployment_checkpoint,
    save_training_checkpoint,
    utc_timestamp,
    validate_resume_checkpoint,
    write_csv,
    write_json,
    write_training_plot,
)
from mlx.modes.video_anomaly_detection.data import (
    VideoClipDataset,
    collate_clip_samples,
)
from mlx.modes.video_anomaly_detection.models import build_video_anomaly_model
from mlx.modes.video_anomaly_detection.requests import TrainVideoAnomalyRequest


TRAINING_COLUMNS = [
    "epoch",
    "train_loss",
    "train_svdd_loss",
    "val_loss",
    "val_score_mean",
    "val_score_std",
    "learning_rate",
    "train_score_mean",
    "train_score_std",
    "val_score_p50",
    "val_score_p95",
]


@torch.no_grad()
def initialize_svdd_center(model, loader, device: str, eps: float = 0.1) -> torch.Tensor:
    was_training = model.training
    model.eval()
    total = torch.zeros_like(model.svdd_head.center, device=device)
    count = 0
    for clips, _, _ in loader:
        embeddings = model(clips.to(device)).svdd_embedding
        total += embeddings.sum(dim=0)
        count += embeddings.shape[0]
    if was_training:
        model.train()
    if count == 0:
        raise MLXUserError("Cannot initialize the Deep SVDD center from an empty normal training loader.")
    center = total / count
    near_zero = center.abs() < eps
    signs = torch.where(center < 0, -torch.ones_like(center), torch.ones_like(center))
    center = torch.where(near_zero, signs * eps, center)
    model.svdd_head.center.copy_(center)
    return center


@torch.no_grad()
def collect_scores(model, loader, device: str) -> torch.Tensor:
    model.eval()
    scores = [model(clips.to(device)).anomaly_score.cpu() for clips, _, _ in loader]
    if not scores:
        raise MLXUserError("Cannot evaluate Deep SVDD using an empty clip loader.")
    return torch.cat(scores)


class TrainVideoAnomalyModel:
    def __init__(
        self,
        request: TrainVideoAnomalyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        model_factory=build_video_anomaly_model,
        dataset_factory=VideoClipDataset,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.model_factory = model_factory
        self.dataset_factory = dataset_factory

    def execute(self) -> dict[str, Any]:
        config = self.request.to_config()
        _apply_resume_backbone_compatibility(config)
        _validate_training_config(config)
        seed_everything(config.get("random_seed"))
        paths = resolve_training_paths(config)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        train_dataset = self._dataset(config, "train", normal_only=True)
        val_dataset = self._dataset(config, "val", normal_only=True)
        train_loader = self._loader(train_dataset, config, shuffle=True)
        center_loader = self._loader(train_dataset, config, shuffle=False)
        val_loader = self._loader(val_dataset, config, shuffle=False)
        device = str(config["device"])
        model_config = dict(config)
        if config.get("model_path"):
            model_config["pretrained"] = False
        model = self.model_factory(str(config["model"]), model_config).to(device)
        optimizer = Adam(model.parameters(), lr=float(config.get("lr") or 0.001))

        start_epoch, best, history, resumed = self._prepare_resume(
            model, optimizer, config, paths["training_csv"]
        )
        if not resumed:
            initialize_svdd_center(model, center_loader, device)

        epochs = int(config["epochs"])
        emit(
            self.reporter,
            "info",
            f"Training normal-only video anomaly model for {epochs} epochs on {device}",
        )
        for epoch in range(start_epoch, epochs):
            train_scores, train_loss = self._train_epoch(model, train_loader, optimizer, device)
            val_scores = collect_scores(model, val_loader, device)
            val_loss = float(val_scores.mean().item())
            row = _history_row(
                epoch + 1,
                train_loss,
                train_scores,
                val_scores,
                float(optimizer.param_groups[0]["lr"]),
            )
            history.append(row)
            write_csv(paths["training_csv"], history, TRAINING_COLUMNS)
            improved = val_loss < best
            if improved:
                best = val_loss
            if improved or not bool(config.get("use_best", True)):
                save_deployment_checkpoint(paths["checkpoint"], model, config)
            save_training_checkpoint(
                paths["last_checkpoint"],
                model,
                optimizer,
                config,
                completed_epoch=epoch + 1,
                history=history,
                best_validation_objective=best,
            )
            emit(
                self.reporter,
                "progress",
                f"Epoch {epoch + 1}/{epochs} train_svdd={train_loss:.6f} val_svdd={val_loss:.6f}",
                current=epoch + 1,
                total=epochs,
                payload={"event": "video_anomaly_training_epoch", "metrics": row, "best": best},
            )

        if not paths["checkpoint"].is_file():
            save_deployment_checkpoint(paths["checkpoint"], model, config)
        self._calibrate_checkpoint(paths["checkpoint"], val_loader, config, device)
        # The current model is the resumable last state; calibrate and persist it without
        # disturbing optimizer/history state.
        last_scores = collect_scores(model, val_loader, device)
        model.svdd_head.threshold.copy_(
            quantile_threshold(last_scores, float(config["svdd_quantile"])).to(device)
        )
        completed_epoch = int(history[-1]["epoch"]) if history else start_epoch
        save_training_checkpoint(
            paths["last_checkpoint"],
            model,
            optimizer,
            config,
            completed_epoch=completed_epoch,
            history=history,
            best_validation_objective=best,
        )
        write_training_plot(paths["training_plot"], history)
        threshold = self._checkpoint_threshold(paths["checkpoint"], device)
        metadata = {
            "mode": "video_anomaly_detection",
            "created_at": utc_timestamp(),
            "model": config["model"],
            "backbone_mode": config["backbone_mode"],
            "dataset": str(config["dataset_path"]),
            "normal_only_training": True,
            "train_windows": len(train_dataset),
            "validation_normal_windows": len(val_dataset),
            "epochs_completed": completed_epoch,
            "seed": config.get("random_seed"),
            "device": device,
            "threshold": threshold,
            "threshold_quantile": float(config["svdd_quantile"]),
            "score_type": "squared_euclidean",
            "input_height": int(config["height"]),
            "input_width": int(config["width"]),
            "clip_length": int(config["clip_length"]),
            "frame_stride": int(config["frame_stride"]),
            "svdd_dim": int(config["svdd_dim"]),
            "svdd_hidden_dim": int(config["svdd_hidden_dim"]),
            "pretrained_initialization": bool(
                config.get("pretrained_initialization", config.get("pretrained", False))
            ),
            "checkpoint": paths["checkpoint"].name,
            "last_checkpoint": paths["last_checkpoint"].name,
        }
        if config["backbone_mode"] == "3d":
            provenance = (
                config.get("pretrained_provenance")
                or getattr(model.backbone, "pretrained_provenance", "none")
            )
            if provenance == "none" and metadata["pretrained_initialization"]:
                provenance = (
                    "inflated_partial"
                    if str(config["model"]).startswith("drax")
                    else "inflated_full"
                )
            metadata.update(
                {
                    "backbone_class": type(model.backbone).__name__,
                    "backbone_temporal_kernel_size": int(
                        config["backbone_temporal_kernel_size"]
                    ),
                    "backbone_temporal_stride_policy": "preserve",
                    "backbone_pooling": "adaptive_avg_3d",
                    "pretrained_provenance": provenance,
                }
            )
        else:
            metadata.update(
                {
                    "temporal_model": config["temporal_model"],
                    "temporal_hidden_dim": int(config["temporal_hidden_dim"]),
                    "temporal_embedding_dim": int(config["temporal_embedding_dim"]),
                    "temporal_kernel_size": int(config["temporal_kernel_size"]),
                    "temporal_dropout": float(config["temporal_dropout"]),
                }
            )
        write_json(paths["run_metadata"], metadata)
        emit(
            self.reporter,
            "success",
            f"Training complete; normal-validation threshold={threshold:.6f}",
            payload={"event": "video_anomaly_training_complete", **metadata},
        )
        return {"paths": paths, "history": history, "metadata": metadata}

    def _dataset(self, config: dict[str, Any], split: str, *, normal_only: bool):
        return self.dataset_factory(
            config["dataset_path"],
            split=split,
            clip_length=int(config["clip_length"]),
            frame_stride=int(config["frame_stride"]),
            height=int(config["height"]),
            width=int(config["width"]),
            normal_only=normal_only,
        )

    @staticmethod
    def _loader(dataset, config: dict[str, Any], *, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle,
            num_workers=int(config.get("workers", 0)),
            collate_fn=collate_clip_samples,
        )

    def _prepare_resume(self, model, optimizer, config, training_csv: Path):
        resume = config.get("model_path")
        if not resume:
            write_csv(training_csv, [], TRAINING_COLUMNS)
            return 0, float("inf"), [], False
        path = Path(resume).expanduser()
        if not path.is_file():
            raise MLXUserError(f"Resume checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=config["device"], weights_only=True)
            validate_resume_checkpoint(checkpoint, config)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (
            EOFError,
            OSError,
            KeyError,
            pickle.UnpicklingError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise MLXUserError(f"Could not resume video-anomaly training from '{path}': {exc}") from exc
        history = list(checkpoint.get("history") or [])
        config["pretrained_initialization"] = bool(
            checkpoint.get("pretrained_initialization", False)
        )
        config["pretrained_provenance"] = checkpoint.get(
            "pretrained_provenance", "none"
        )
        completed = int(checkpoint.get("completed_epoch", 0))
        if len(history) != completed:
            raise MLXUserError("Resume checkpoint epoch history is inconsistent.")
        if completed > int(config["epochs"]):
            raise MLXUserError(
                f"Resume checkpoint completed epoch {completed}, exceeding --epochs={config['epochs']}."
            )
        restore_random_state(checkpoint.get("random_state") or {})
        write_csv(training_csv, history, TRAINING_COLUMNS)
        emit(self.reporter, "info", f"Resuming from completed epoch {completed}: {path}")
        return completed, float(checkpoint.get("best_validation_objective", float("inf"))), history, True

    @staticmethod
    def _train_epoch(model, loader, optimizer, device: str):
        model.train()
        scores = []
        for clips, labels, _ in loader:
            if torch.any(labels != 0):
                raise MLXUserError("An anomalous sample reached the normal-only training loader.")
            optimizer.zero_grad()
            batch_scores = model(clips.to(device)).anomaly_score
            loss = batch_scores.mean()
            loss.backward()
            optimizer.step()
            scores.append(batch_scores.detach().cpu())
        all_scores = torch.cat(scores)
        return all_scores, float(all_scores.mean().item())

    def _calibrate_checkpoint(self, path: Path, val_loader, config, device: str) -> None:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        calibrated = self.model_factory(
            str(config["model"]), {**config, "pretrained": False}
        ).to(device)
        calibrated.load_state_dict(checkpoint["state_dict"])
        scores = collect_scores(calibrated, val_loader, device)
        threshold = quantile_threshold(scores, float(config["svdd_quantile"])).to(device)
        calibrated.svdd_head.threshold.copy_(threshold)
        save_deployment_checkpoint(path, calibrated, config)

    @staticmethod
    def _checkpoint_threshold(path: Path, device: str) -> float:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        threshold = checkpoint.get("svdd_threshold")
        if threshold is None:
            raise MLXUserError("Final checkpoint threshold calibration failed.")
        return float(threshold)


def _history_row(epoch, train_loss, train_scores, val_scores, learning_rate):
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_svdd_loss": train_loss,
        "val_loss": float(val_scores.mean().item()),
        "val_score_mean": float(val_scores.mean().item()),
        "val_score_std": float(val_scores.std(unbiased=False).item()),
        "learning_rate": learning_rate,
        "train_score_mean": float(train_scores.mean().item()),
        "train_score_std": float(train_scores.std(unbiased=False).item()),
        "val_score_p50": float(torch.quantile(val_scores, 0.5).item()),
        "val_score_p95": float(torch.quantile(val_scores, 0.95).item()),
    }


def _validate_training_config(config: dict[str, Any]) -> None:
    if not config.get("model"):
        raise MLXUserError("Video anomaly training requires --model.")
    from mlx.modes.image_classification.models import model_family_for

    if model_family_for(str(config["model"])) != "standard":
        raise MLXUserError(
            "Video anomaly detection supports standard backbones only; "
            "Siamese models are excluded."
        )
    if int(config.get("epochs", 0)) < 1:
        raise MLXUserError("--epochs must be at least 1.")
    if int(config.get("batch_size", 0)) < 1:
        raise MLXUserError("--batch-size must be at least 1.")
    if float(config.get("lr") or 0) <= 0:
        raise MLXUserError("--lr must be greater than zero.")
    backbone_mode = str(config.get("backbone_mode", "3d"))
    if backbone_mode not in {"3d", "frame-2d"}:
        raise MLXUserError("--backbone-mode must be '3d' or 'frame-2d'.")
    if backbone_mode == "3d":
        kernel = int(config.get("backbone_temporal_kernel_size", 3))
        if kernel < 1 or kernel % 2 == 0:
            raise MLXUserError(
                "--backbone-temporal-kernel-size must be a positive odd integer."
            )
        if int(config.get("clip_length", 0)) < kernel:
            raise MLXUserError(
                "--clip-length must be at least --backbone-temporal-kernel-size."
            )
    quantile = float(config.get("svdd_quantile", 0.95))
    if not 0 < quantile < 1:
        raise MLXUserError("--svdd-quantile must be strictly between zero and one.")


def _apply_resume_backbone_compatibility(config: dict[str, Any]) -> None:
    resume = config.get("model_path")
    explicit = set(config.get("_explicit_options") or ())
    if not resume or "backbone_mode" in explicit:
        return
    path = Path(resume).expanduser()
    if not path.is_file():
        return
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError):
        return  # The normal resume loader supplies the contextual user-facing error.
    if checkpoint.get("mode") == "video_anomaly_detection":
        stored_mode = str(checkpoint.get("backbone_mode", "frame-2d"))
        temporal_options = {
            "temporal_model",
            "temporal_hidden_dim",
            "temporal_embedding_dim",
            "temporal_kernel_size",
            "temporal_dropout",
        }
        if stored_mode == "3d" and explicit & temporal_options:
            raise MLXUserError(
                "The resume checkpoint uses a 3D backbone; temporal-CNN options cannot alter "
                "its stored architecture."
            )
        config["backbone_mode"] = stored_mode


__all__ = [
    "TRAINING_COLUMNS",
    "TrainVideoAnomalyModel",
    "collect_scores",
    "initialize_svdd_center",
]
