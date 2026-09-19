from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mlx.core.artifacts import atomic_torch_save
from mlx.core.exceptions import MLXUserError
from mlx.core.random import capture_random_state, restore_random_state
from mlx.modes.saliency_mapping.models import DEFAULT_MODEL, build_saliency_model


def training_paths(config: dict[str, Any], *, model_name: str) -> dict[str, Path]:
    value = config.get("output_path")
    if not value:
        raise MLXUserError("Saliency training requires --output.")
    output = Path(value).expanduser()
    if output.suffix.lower() in {".pt", ".pth"} and not output.is_dir():
        root = output.parent / f"{output.stem}-research"
        best = output
    else:
        root = output
        best = root / f"{model_name}.pth"
    return {
        "output_dir": root,
        "checkpoint_path": best,
        "last_checkpoint_path": best.with_name(f"{best.stem}.last{best.suffix}"),
        "training_csv_path": root / "training.csv",
        "training_curves_path": root / "training_curves.png",
        "training_config_path": root / "training_config.json",
    }


def checkpoint_payload(model, *, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": "saliency_mapping",
        "model_name": model_name,
        "input_size": tuple(config.get("input_size", (256, 256))),
        "transform": str(config.get("transform", "resize")),
        "colored": bool(config.get("colored", True)),
        "output_channels": 1,
        "activation": "sigmoid",
        "model_config": dict(config),
        "state_dict": model.state_dict(),
    }


def save_checkpoint(path: Path, model, *, model_name: str, config: dict[str, Any]) -> None:
    atomic_torch_save(checkpoint_payload(model, model_name=model_name, config=config), path)


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    model_name: str,
    config: dict[str, Any],
    completed_epoch: int,
    best_validation_mae: float,
    history: list[dict[str, Any]],
) -> None:
    payload = checkpoint_payload(model, model_name=model_name, config=config)
    payload.update(
        {
            "training_state_version": 1,
            "completed_epoch": int(completed_epoch),
            "best_validation_mae": float(best_validation_mae),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": list(history),
            "random_state": capture_random_state(),
        }
    )
    atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model,
    optimizer,
    *,
    model_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = _load(path, config.get("device", "cpu"), "resume")
    if checkpoint.get("family") != "saliency_mapping" or checkpoint.get("training_state_version") != 1:
        raise MLXUserError(f"Checkpoint '{path}' is not a resumable saliency checkpoint.")
    expected = {
        "model_name": model_name,
        "input_size": tuple(config.get("input_size", (256, 256))),
        "transform": str(config.get("transform", "resize")),
        "colored": bool(config.get("colored", True)),
    }
    actual = {
        "model_name": checkpoint.get("model_name"),
        "input_size": tuple(checkpoint.get("input_size", ())),
        "transform": checkpoint.get("transform", "resize"),
        "colored": bool(checkpoint.get("colored", True)),
    }
    if actual != expected:
        raise MLXUserError(
            f"Saliency resume checkpoint metadata does not match the request: expected {expected}, got {actual}."
        )
    try:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Saliency resume checkpoint '{path}' is incompatible: {exc}") from exc
    restore_random_state(checkpoint.get("random_state") or {})
    return {
        "completed_epoch": int(checkpoint.get("completed_epoch", 0)),
        "best_validation_mae": float(checkpoint.get("best_validation_mae", float("inf"))),
        "history": list(checkpoint.get("history") or []),
    }


def load_checkpoint_bundle(config: dict[str, Any]):
    path = config.get("model_path")
    if not path:
        raise MLXUserError("This saliency action requires --model-path.")
    checkpoint = _load(path, config.get("device", "cpu"), "model")
    if checkpoint.get("family") != "saliency_mapping" or "state_dict" not in checkpoint:
        raise MLXUserError(
            f"Checkpoint '{path}' is not an MLX saliency-mapping checkpoint."
        )
    runtime = dict(config)
    runtime["colored"] = bool(checkpoint.get("colored", True))
    runtime["input_size"] = tuple(checkpoint.get("input_size", (256, 256)))
    runtime["transform"] = str(checkpoint.get("transform", "resize"))
    model_name = config.get("model") or checkpoint.get("model_name") or DEFAULT_MODEL
    model = build_saliency_model(model_name, runtime)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise MLXUserError(f"Saliency checkpoint '{path}' is incompatible with '{model_name}': {exc}") from exc
    return model, {
        "checkpoint_path": Path(path),
        "model_name": model_name,
        "input_size": runtime["input_size"],
        "transform": runtime["transform"],
        "colored": runtime["colored"],
        "output_channels": 1,
    }


def _load(path: str | Path, device: str, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise MLXUserError(f"Saliency {label} checkpoint not found: {resolved}")
    try:
        return torch.load(resolved, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Could not load saliency checkpoint '{resolved}': {exc}") from exc


__all__ = [
    "checkpoint_payload",
    "load_checkpoint_bundle",
    "load_training_checkpoint",
    "save_checkpoint",
    "save_training_checkpoint",
    "training_paths",
]
