from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.models import build_video_anomaly_model


CHECKPOINT_VERSION = 2
SCORE_TYPE = "squared_euclidean"


def artifact_stem(config: dict[str, Any]) -> str:
    if config.get("backbone_mode", "3d") == "3d":
        fusion = (
            f"-{config.get('drax_fusion_mode', 'average')}"
            if str(config["model"]).startswith("drax")
            else ""
        )
        return f"{config['model']}{fusion}-3d-svdd"
    return f"{config['model']}-{config.get('temporal_model', 'tcn')}-svdd"


def resolve_training_paths(config: dict[str, Any]) -> dict[str, Path]:
    if not config.get("output_path"):
        raise MLXUserError("Training requires --output pointing to an artifact directory.")
    output_dir = Path(config["output_path"]).expanduser()
    stem = artifact_stem(config)
    return {
        "output_dir": output_dir,
        "checkpoint": output_dir / f"{stem}.pth",
        "last_checkpoint": output_dir / f"{stem}.last.pth",
        "training_csv": output_dir / "training.csv",
        "run_metadata": output_dir / "run_metadata.json",
        "training_plot": output_dir / "training_history.png",
    }


def model_metadata(model, config: dict[str, Any]) -> dict[str, Any]:
    backbone_mode = str(getattr(model, "backbone_mode", config.get("backbone_mode", "3d")))
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mode": "video_anomaly_detection",
        "model_name": str(config["model"]),
        "model_family": "standard",
        "backbone_mode": backbone_mode,
        "backbone_feature_dim": int(model.backbone_feature_dim),
        "pretrained_initialization": bool(
            config.get("pretrained_initialization", config.get("pretrained", False))
        ),
        "input_height": int(config.get("height", 224)),
        "input_width": int(config.get("width", 224)),
        "clip_length": int(config.get("clip_length", 16)),
        "frame_stride": int(config.get("frame_stride", 1)),
        "svdd_dim": int(config.get("svdd_dim", 128)),
        "svdd_hidden_dim": int(config.get("svdd_hidden_dim", 256)),
        "svdd_quantile": float(config.get("svdd_quantile", 0.95)),
        "svdd_center": model.svdd_head.center.detach().cpu(),
        "svdd_threshold": _finite_float_or_none(model.svdd_head.threshold),
        "score_type": SCORE_TYPE,
        "label_semantics": {"0": "normal", "1": "anomaly"},
        "drax_fusion_mode": str(config.get("drax_fusion_mode", "average")),
        "mlx_version": _mlx_version(),
    }
    if backbone_mode == "3d":
        backbone = model.backbone
        provenance = str(
            config.get("pretrained_provenance") or backbone.pretrained_provenance
        )
        if provenance == "none" and metadata["pretrained_initialization"]:
            provenance = (
                "inflated_partial"
                if str(config["model"]).startswith("drax")
                else "inflated_full"
            )
        metadata.update(
            {
                "backbone_class": type(backbone).__name__,
                "backbone_temporal_kernel_size": int(backbone.temporal_kernel_size),
                "backbone_temporal_stride_policy": str(backbone.temporal_stride_policy),
                "backbone_pooling": str(backbone.pooling),
                "pretrained_provenance": provenance,
                "clip_embedding_dim": int(backbone.feature_dim),
            }
        )
    else:
        metadata.update(
            {
                "backbone_class": type(model.frame_backbone.backbone).__name__,
                "temporal_model": str(config.get("temporal_model", "tcn")),
                "temporal_hidden_dim": int(config.get("temporal_hidden_dim", 256)),
                "temporal_embedding_dim": int(config.get("temporal_embedding_dim", 128)),
                "temporal_kernel_size": int(config.get("temporal_kernel_size", 3)),
                "temporal_dropout": float(config.get("temporal_dropout", 0.0)),
                "clip_embedding_dim": int(config.get("temporal_embedding_dim", 128)),
            }
        )
    return metadata


def checkpoint_payload(model, config: dict[str, Any]) -> dict[str, Any]:
    return {
        **model_metadata(model, config),
        "state_dict": model.state_dict(),
    }


def save_deployment_checkpoint(path: Path, model, config: dict[str, Any]) -> None:
    atomic_torch_save(checkpoint_payload(model, config), path)


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    config: dict[str, Any],
    *,
    completed_epoch: int,
    history: list[dict[str, Any]],
    best_validation_objective: float,
) -> None:
    payload = checkpoint_payload(model, config)
    payload.update(
        {
            "training_state_version": 1,
            "completed_epoch": int(completed_epoch),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": list(history),
            "best_validation_objective": float(best_validation_objective),
            "random_state": capture_random_state(),
        }
    )
    atomic_torch_save(payload, path)


def load_video_anomaly_checkpoint(
    model_path: str | Path,
    *,
    device: str = "cpu",
    model_name: str | None = None,
    model_factory=build_video_anomaly_model,
):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise MLXUserError(f"Video-anomaly checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Could not load video-anomaly checkpoint '{path}': {exc}") from exc
    if checkpoint.get("mode") != "video_anomaly_detection" or "state_dict" not in checkpoint:
        raise MLXUserError(
            f"Checkpoint '{path}' is not a metadata-complete video-anomaly checkpoint."
        )
    checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
    if checkpoint_version > CHECKPOINT_VERSION:
        raise MLXUserError(
            f"Checkpoint '{path}' uses unsupported version {checkpoint_version}; "
            f"this MLX build supports through version {CHECKPOINT_VERSION}."
        )
    stored_model = checkpoint.get("model_name")
    if model_name and model_name != stored_model:
        raise MLXUserError(
            f"Checkpoint backbone '{stored_model}' does not match requested model '{model_name}'."
        )
    stored_center = checkpoint.get("svdd_center")
    if stored_center is None:
        raise MLXUserError(f"Checkpoint '{path}' has no Deep SVDD center.")
    try:
        if not torch.all(torch.isfinite(torch.as_tensor(stored_center))):
            raise ValueError("the stored Deep SVDD center contains non-finite values")
        config = config_from_checkpoint(checkpoint)
        config["device"] = device
        config["pretrained"] = False
        model = model_factory(str(stored_model), config)
        stored_class = checkpoint.get("backbone_class")
        if config["backbone_mode"] == "3d" and stored_class:
            actual_class = type(model.backbone).__name__
            if stored_class != actual_class:
                raise ValueError(
                    f"stored backbone class '{stored_class}' reconstructed as '{actual_class}'"
                )
        if int(checkpoint.get("backbone_feature_dim", model.backbone_feature_dim)) != int(
            model.backbone_feature_dim
        ):
            raise ValueError("the stored backbone feature dimension does not match the model")
        model.load_state_dict(checkpoint["state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise MLXUserError(f"Checkpoint '{path}' is incompatible with its architecture metadata: {exc}") from exc
    return model.to(device), checkpoint, config


def config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    backbone_mode = str(checkpoint.get("backbone_mode", "frame-2d"))
    config = {
        "model": checkpoint["model_name"],
        "height": int(checkpoint["input_height"]),
        "width": int(checkpoint["input_width"]),
        "clip_length": int(checkpoint["clip_length"]),
        "frame_stride": int(checkpoint["frame_stride"]),
        "backbone_mode": backbone_mode,
        "svdd_dim": int(checkpoint["svdd_dim"]),
        "svdd_hidden_dim": int(checkpoint["svdd_hidden_dim"]),
        "svdd_quantile": float(checkpoint["svdd_quantile"]),
        "drax_fusion_mode": checkpoint.get("drax_fusion_mode", "average"),
    }
    if backbone_mode == "3d":
        config["backbone_temporal_kernel_size"] = int(
            checkpoint["backbone_temporal_kernel_size"]
        )
    else:
        config.update(
            {
                "temporal_model": checkpoint["temporal_model"],
                "temporal_hidden_dim": int(checkpoint["temporal_hidden_dim"]),
                "temporal_embedding_dim": int(checkpoint["temporal_embedding_dim"]),
                "temporal_kernel_size": int(checkpoint["temporal_kernel_size"]),
                "temporal_dropout": float(checkpoint["temporal_dropout"]),
            }
        )
    return config


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if checkpoint.get("training_state_version") != 1:
        raise MLXUserError("The selected checkpoint is not a resumable video-anomaly checkpoint.")
    backbone_mode = str(config.get("backbone_mode", "3d"))
    current = {
        "model_name": config["model"],
        "input_height": int(config["height"]),
        "input_width": int(config["width"]),
        "clip_length": int(config["clip_length"]),
        "frame_stride": int(config["frame_stride"]),
        "backbone_mode": backbone_mode,
        "svdd_dim": int(config["svdd_dim"]),
        "svdd_hidden_dim": int(config["svdd_hidden_dim"]),
        "svdd_quantile": float(config["svdd_quantile"]),
        "drax_fusion_mode": config.get("drax_fusion_mode", "average"),
    }
    if backbone_mode == "3d":
        current["backbone_temporal_kernel_size"] = int(
            config.get("backbone_temporal_kernel_size", 3)
        )
    else:
        current.update(
            {
                "temporal_model": config["temporal_model"],
                "temporal_hidden_dim": int(config["temporal_hidden_dim"]),
                "temporal_embedding_dim": int(config["temporal_embedding_dim"]),
                "temporal_kernel_size": int(config["temporal_kernel_size"]),
                "temporal_dropout": float(config.get("temporal_dropout", 0.0)),
            }
        )
    stored_mode = str(checkpoint.get("backbone_mode", "frame-2d"))
    checkpoint = {**checkpoint, "backbone_mode": stored_mode}
    mismatches = [key for key, value in current.items() if checkpoint.get(key) != value]
    if mismatches:
        raise MLXUserError(
            "Resume checkpoint architecture does not match the request: " + ", ".join(mismatches)
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(_json_value(value), output, indent=2, sort_keys=True)
        output.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(_json_value(row), sort_keys=True) + "\n")


def write_training_plot(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Video Anomaly Deep SVDD Loss", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["learning_rate"] for row in history])
    axes[1].set(title="Learning Rate", xlabel="Epoch", ylabel="Learning rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_random_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    numpy_state = state.get("numpy")
    if numpy_state:
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _finite_float_or_none(value: torch.Tensor) -> float | None:
    return float(value.item()) if torch.isfinite(value) else None


def _mlx_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mlx")
    except PackageNotFoundError:  # package metadata is optional in source checkouts
        return "0.1.0"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


__all__ = [
    "SCORE_TYPE",
    "artifact_stem",
    "checkpoint_payload",
    "config_from_checkpoint",
    "load_video_anomaly_checkpoint",
    "model_metadata",
    "resolve_training_paths",
    "restore_random_state",
    "save_deployment_checkpoint",
    "save_training_checkpoint",
    "sha256_file",
    "utc_timestamp",
    "validate_resume_checkpoint",
    "write_csv",
    "write_json",
    "write_jsonl",
    "write_training_plot",
]
