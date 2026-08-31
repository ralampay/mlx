from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mlx.core.artifacts import (
    atomic_torch_save,
    write_json_atomic,
)
from mlx.core.exceptions import MLXUserError
from mlx.core.random import capture_random_state
from mlx.modes.image_recognition_oc.algorithms import (
    DEFAULT_ALGORITHM_REGISTRY,
    OneClassAlgorithm,
    OneClassAlgorithmRegistry,
)
from mlx.modes.image_recognition_oc.backbones import backbone_class_name


CHECKPOINT_VERSION = 1


def artifact_stem(config: dict[str, Any]) -> str:
    return f"{config['backbone']}-{config['model']}"


def resolve_training_paths(config: dict[str, Any]) -> dict[str, Path]:
    if not config.get("output_path"):
        raise MLXUserError("Training requires --output pointing to an artifact directory.")
    output_dir = Path(str(config["output_path"])).expanduser()
    stem = artifact_stem(config)
    return {
        "output_dir": output_dir,
        "checkpoint": output_dir / f"{stem}.pth",
        "last_checkpoint": output_dir / f"{stem}.last.pth",
        "training_csv": output_dir / "training.csv",
        "training_plot": output_dir / "training_history.png",
        "run_metadata": output_dir / "run_metadata.json",
    }


def model_metadata(model, config: dict[str, Any], algorithm: OneClassAlgorithm) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mode": "image_recognition_oc",
        "model_name": str(config["model"]),
        "backbone_name": str(config["backbone"]),
        "backbone_class": backbone_class_name(model.backbone),
        "backbone_feature_dim": int(model.backbone_feature_dim),
        "input_height": int(config.get("height", 224)),
        "input_width": int(config.get("width", 224)),
        "colored": bool(config.get("colored", True)),
        "pretrained_initialization": bool(
            config.get("pretrained_initialization", config.get("pretrained", False))
        ),
        "drax_fusion_mode": str(config.get("drax_fusion_mode", "average")),
        "label_semantics": {"0": "normal", "1": "anomaly"},
        **algorithm.checkpoint_metadata(model, config),
    }


def checkpoint_payload(model, config: dict[str, Any], algorithm: OneClassAlgorithm):
    return {
        **model_metadata(model, config, algorithm),
        "state_dict": model.state_dict(),
    }


def save_deployment_checkpoint(
    path: Path,
    model,
    config: dict[str, Any],
    algorithm: OneClassAlgorithm,
) -> None:
    atomic_torch_save(checkpoint_payload(model, config, algorithm), path)


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    config: dict[str, Any],
    algorithm: OneClassAlgorithm,
    *,
    completed_epoch: int,
    history: list[dict[str, Any]],
    best_validation_objective: float,
) -> None:
    payload = checkpoint_payload(model, config, algorithm)
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


def update_checkpoint_model(
    path: Path,
    checkpoint: dict[str, Any],
    model,
    config: dict[str, Any],
    algorithm: OneClassAlgorithm,
) -> None:
    atomic_torch_save(
        {**checkpoint, **checkpoint_payload(model, config, algorithm)},
        path,
    )


def load_raw_checkpoint(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser()
    if not target.is_file():
        raise MLXUserError(f"One-class image checkpoint not found: {target}")
    try:
        value = torch.load(target, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Could not load one-class image checkpoint '{target}': {exc}") from exc
    required = {"checkpoint_version", "model_name", "backbone_name", "state_dict"}
    if (
        not isinstance(value, dict)
        or value.get("mode") != "image_recognition_oc"
        or not required.issubset(value)
    ):
        raise MLXUserError(
            f"Checkpoint '{target}' is not a metadata-complete image-recognition-oc checkpoint."
        )
    try:
        version = int(value["checkpoint_version"])
    except (TypeError, ValueError) as exc:
        raise MLXUserError(
            f"Checkpoint '{target}' has an invalid checkpoint version."
        ) from exc
    if version < 1 or version > CHECKPOINT_VERSION:
        raise MLXUserError(
            f"Checkpoint '{target}' uses unsupported version {version}; "
            f"this MLX build supports version {CHECKPOINT_VERSION}."
        )
    return value


def checkpoint_config(
    checkpoint: dict[str, Any], algorithm: OneClassAlgorithm
) -> dict[str, Any]:
    return {
        "model": checkpoint["model_name"],
        "backbone": checkpoint["backbone_name"],
        "height": int(checkpoint["input_height"]),
        "width": int(checkpoint["input_width"]),
        "colored": bool(checkpoint["colored"]),
        "pretrained": False,
        "drax_fusion_mode": checkpoint.get("drax_fusion_mode", "average"),
        **algorithm.config_from_checkpoint(checkpoint),
    }


def load_image_one_class_checkpoint(
    model_path: str | Path,
    *,
    device: str,
    model_name: str | None = None,
    backbone_name: str | None = None,
    registry: OneClassAlgorithmRegistry = DEFAULT_ALGORITHM_REGISTRY,
):
    checkpoint = load_raw_checkpoint(model_path)
    if model_name and model_name != checkpoint["model_name"]:
        raise MLXUserError(
            f"Checkpoint model '{checkpoint['model_name']}' does not match requested model '{model_name}'."
        )
    if backbone_name and backbone_name != checkpoint["backbone_name"]:
        raise MLXUserError(
            f"Checkpoint backbone '{checkpoint['backbone_name']}' does not match requested backbone "
            f"'{backbone_name}'."
        )
    algorithm = registry.get(str(checkpoint["model_name"]))
    config = checkpoint_config(checkpoint, algorithm)
    try:
        model = algorithm.build_model(str(checkpoint["backbone_name"]), config)
        if checkpoint["backbone_class"] != backbone_class_name(model.backbone):
            raise ValueError("stored backbone class does not match the reconstructed model")
        if int(checkpoint["backbone_feature_dim"]) != int(model.backbone_feature_dim):
            raise ValueError("stored backbone feature dimension does not match the reconstructed model")
        model.load_state_dict(checkpoint["state_dict"])
        algorithm.validate_loaded_model(model, checkpoint)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise MLXUserError(
            f"Checkpoint '{Path(model_path).expanduser()}' is incompatible with its architecture metadata: {exc}"
        ) from exc
    threshold = checkpoint.get("threshold")
    if threshold is not None and not torch.isfinite(torch.tensor(float(threshold))):
        raise MLXUserError(f"Checkpoint '{model_path}' contains a non-finite threshold.")
    return model.to(device), checkpoint, config, algorithm


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    algorithm: OneClassAlgorithm,
) -> None:
    if checkpoint.get("training_state_version") != 1:
        raise MLXUserError("The selected checkpoint is not a resumable one-class image checkpoint.")
    expected = {
        "model_name": str(config["model"]),
        "backbone_name": str(config["backbone"]),
        "input_height": int(config["height"]),
        "input_width": int(config["width"]),
        "colored": bool(config["colored"]),
        "drax_fusion_mode": str(config.get("drax_fusion_mode", "average")),
        **algorithm.resume_signature(config),
    }
    mismatches = [
        name
        for name, value in expected.items()
        if checkpoint.get(name) != value
    ]
    if mismatches:
        raise MLXUserError(
            "Resume checkpoint configuration does not match: " + ", ".join(mismatches) + "."
        )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_training_plot(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
    axis.set(title="One-Class Image Training", xlabel="Epoch", ylabel="Mean anomaly score")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


write_json = write_json_atomic


__all__ = [
    "CHECKPOINT_VERSION",
    "artifact_stem",
    "checkpoint_config",
    "load_image_one_class_checkpoint",
    "load_raw_checkpoint",
    "model_metadata",
    "resolve_training_paths",
    "save_deployment_checkpoint",
    "save_training_checkpoint",
    "update_checkpoint_model",
    "utc_timestamp",
    "validate_resume_checkpoint",
    "write_json",
    "write_training_plot",
]
