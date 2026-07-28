from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.models import DEFAULT_MODEL, build_segmentation_model


def resolve_model_name(config: dict[str, Any]) -> str:
    return config.get("model") or DEFAULT_MODEL


def resolve_train_output_path(config: dict[str, Any]) -> Path:
    return resolve_train_output_paths(config, model_name=resolve_model_name(config))[
        "checkpoint_path"
    ]


def resolve_train_output_paths(
    config: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Path]:
    output_value = config.get("output_path")
    if not output_value:
        raise MLXUserError(
            "Training requires --output pointing to an artifact directory or checkpoint file."
        )
    output_path = Path(output_value).expanduser()
    is_existing_directory = output_path.exists() and output_path.is_dir()
    is_checkpoint_file = not is_existing_directory and output_path.suffix.lower() in {
        ".pt",
        ".pth",
    }
    if is_checkpoint_file:
        suffix = output_path.suffix
        stem = output_path.stem
        artifact_dir = output_path.parent / f"{stem}-research"
        return {
            "output_dir": artifact_dir,
            "checkpoint_path": output_path,
            "dice_checkpoint_path": output_path.with_name(f"{stem}.best-dice{suffix}"),
            "last_checkpoint_path": output_path.with_name(f"{stem}.last{suffix}"),
            "training_csv_path": artifact_dir / "training.csv",
            "training_curves_path": artifact_dir / "training_curves.png",
            "training_config_path": artifact_dir / "training_config.json",
        }
    return {
        "output_dir": output_path,
        "checkpoint_path": output_path / f"{model_name}.pth",
        "dice_checkpoint_path": output_path / f"{model_name}.best-dice.pth",
        "last_checkpoint_path": output_path / f"{model_name}.last.pth",
        "training_csv_path": output_path / "training.csv",
        "training_curves_path": output_path / "training_curves.png",
        "training_config_path": output_path / "training_config.json",
    }


def resolve_class_names(
    config: dict[str, Any],
    num_classes: int,
    *,
    checkpoint_names: list[str] | None = None,
) -> list[str]:
    configured = config.get("class_names")
    if isinstance(configured, str):
        names = [value.strip() for value in configured.split(",")]
    elif configured:
        names = [str(value).strip() for value in configured]
    elif checkpoint_names:
        names = list(checkpoint_names)
    elif num_classes == 2:
        names = ["background", "foreground"]
    else:
        names = [f"class_{index}" for index in range(num_classes)]
    if len(names) != num_classes:
        raise MLXUserError(
            f"Expected {num_classes} class names but received {len(names)}. "
            "Pass a comma-separated --class-names value matching --num-classes."
        )
    if any(not name for name in names):
        raise MLXUserError("--class-names cannot contain empty names.")
    if len(set(names)) != len(names):
        raise MLXUserError("--class-names must contain unique names.")
    return names


def default_palette(num_classes: int) -> list[list[int]]:
    if num_classes <= 2:
        return [[0, 0, 0], [255, 80, 80]]
    base = [
        [0, 0, 0],
        [255, 80, 80],
        [80, 180, 255],
        [120, 220, 120],
        [255, 210, 90],
        [190, 120, 255],
    ]
    if num_classes <= len(base):
        return base[:num_classes]
    palette = list(base)
    while len(palette) < num_classes:
        index = len(palette)
        palette.append(
            [
                int((37 * index) % 255),
                int((97 * index) % 255),
                int((173 * index) % 255),
            ]
        )
    return palette


def checkpoint_payload(
    model,
    *,
    model_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    num_classes = int(config.get("num_classes", 2))
    return {
        "class_names": resolve_class_names(config, num_classes),
        "colored": bool(config.get("colored", True)),
        "family": "segmentation",
        "input_size": tuple(config.get("input_size", (256, 256))),
        "mask_threshold": float(config.get("mask_threshold", 0.5)),
        "model_config": dict(config),
        "model_name": model_name,
        "num_classes": num_classes,
        "palette": config.get("palette") or default_palette(num_classes),
        "state_dict": model.state_dict(),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model,
    *,
    model_name: str,
    config: dict[str, Any],
) -> None:
    _atomic_torch_save(
        checkpoint_payload(model, model_name=model_name, config=config),
        checkpoint_path,
    )


def save_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    *,
    model_name: str,
    config: dict[str, Any],
    completed_epoch: int,
    best_val_loss: float,
    best_foreground_dice: float,
    history: list[dict[str, Any]],
) -> None:
    payload = checkpoint_payload(model, model_name=model_name, config=config)
    payload.update(
        {
            "training_state_version": 1,
            "completed_epoch": int(completed_epoch),
            "best_val_loss": float(best_val_loss),
            "best_foreground_dice": float(best_foreground_dice),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": list(history),
            "random_state": _capture_random_state(),
        }
    )
    _atomic_torch_save(payload, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer,
    *,
    model_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise MLXUserError(f"Resume checkpoint not found: {path}")
    try:
        checkpoint = torch.load(
            path,
            map_location=config.get("device", "cpu"),
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Could not load segmentation resume checkpoint '{path}': {exc}") from exc
    if checkpoint.get("training_state_version") != 1:
        raise MLXUserError(
            f"Checkpoint '{path}' is not a resumable segmentation training checkpoint. "
            "Use the generated '.last' checkpoint."
        )
    expected_num_classes = int(config.get("num_classes", 2))
    expected_names = resolve_class_names(config, expected_num_classes)
    checks = (
        ("model", checkpoint.get("model_name"), model_name),
        ("class count", int(checkpoint.get("num_classes", 0)), expected_num_classes),
        ("class names", list(checkpoint.get("class_names") or []), expected_names),
        (
            "input size",
            tuple(checkpoint.get("input_size") or ()),
            tuple(config.get("input_size", (256, 256))),
        ),
        ("color mode", bool(checkpoint.get("colored", True)), bool(config.get("colored", True))),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise MLXUserError(
                f"Resume checkpoint {label} '{actual}' does not match requested '{expected}'."
            )
    try:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Resume checkpoint '{path}' is incompatible: {exc}") from exc
    history = list(checkpoint.get("history") or [])
    completed_epoch = int(checkpoint.get("completed_epoch", 0))
    if len(history) != completed_epoch:
        raise MLXUserError(f"Resume checkpoint '{path}' has inconsistent epoch history.")
    _restore_random_state(checkpoint.get("random_state") or {})
    return {
        "completed_epoch": completed_epoch,
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "best_foreground_dice": float(
            checkpoint.get("best_foreground_dice", float("-inf"))
        ),
        "history": history,
    }


def load_checkpoint_bundle(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    model_path = config.get("model_path")
    if not model_path:
        raise MLXUserError("This action requires --model-path pointing to a checkpoint.")

    try:
        checkpoint = torch.load(
            model_path,
            map_location=config.get("device", "cpu"),
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Could not load segmentation checkpoint '{model_path}': {exc}") from exc
    if "state_dict" not in checkpoint:
        raise MLXUserError(
            f"Checkpoint '{model_path}' does not include segmentation metadata. "
            "Re-train the model with the segmentation mode."
        )
    if checkpoint.get("family") not in (None, "segmentation"):
        raise MLXUserError(
            f"Checkpoint '{model_path}' belongs to family '{checkpoint.get('family')}', not segmentation."
        )

    runtime_config = dict(config)
    runtime_config["colored"] = checkpoint.get("colored", runtime_config.get("colored", True))
    runtime_config["input_size"] = tuple(
        checkpoint.get("input_size", runtime_config.get("input_size", (256, 256)))
    )
    runtime_config["num_classes"] = int(
        checkpoint.get("num_classes", runtime_config.get("num_classes", 2))
    )

    model_name = config.get("model") or checkpoint.get("model_name") or DEFAULT_MODEL
    model = build_segmentation_model(
        model_name,
        runtime_config,
        num_classes=runtime_config["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])

    metadata = {
        "checkpoint_path": Path(model_path),
        "class_names": checkpoint.get("class_names"),
        "colored": runtime_config["colored"],
        "input_size": runtime_config["input_size"],
        "mask_threshold": float(checkpoint.get("mask_threshold", config.get("mask_threshold", 0.5))),
        "model_name": model_name,
        "num_classes": runtime_config["num_classes"],
        "palette": checkpoint.get("palette") or default_palette(runtime_config["num_classes"]),
    }
    return model, metadata


def _atomic_torch_save(payload: dict[str, Any], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _capture_random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
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
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(tuple(state["python"]))
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
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def compute_pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total else 0.0


def compute_dice_score(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    scores = []
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    for class_index in range(num_classes):
        pred_mask = predictions == class_index
        target_mask = targets == class_index
        intersection = float((pred_mask & target_mask).sum().item())
        denominator = float(pred_mask.sum().item() + target_mask.sum().item())
        if denominator == 0:
            scores.append(1.0)
            continue
        scores.append((2.0 * intersection) / denominator)
    return float(np.mean(scores)) if scores else 0.0


def compute_mean_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    scores = []
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    for class_index in range(num_classes):
        pred_mask = predictions == class_index
        target_mask = targets == class_index
        intersection = float((pred_mask & target_mask).sum().item())
        union = float((pred_mask | target_mask).sum().item())
        if union == 0:
            scores.append(1.0)
            continue
        scores.append(intersection / union)
    return float(np.mean(scores)) if scores else 0.0
