from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.models import DEFAULT_MODEL, build_segmentation_model


def resolve_model_name(config: dict[str, Any]) -> str:
    return config.get("model") or DEFAULT_MODEL


def resolve_train_output_path(config: dict[str, Any]) -> Path:
    output_path = config.get("output_path")
    if not output_path:
        raise MLXUserError("Training requires --output pointing to the model file to write.")
    return Path(output_path).expanduser()


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
    return {
        "colored": bool(config.get("colored", True)),
        "family": "segmentation",
        "input_size": tuple(config.get("input_size", (256, 256))),
        "mask_threshold": float(config.get("mask_threshold", 0.5)),
        "model_name": model_name,
        "num_classes": int(config.get("num_classes", 2)),
        "palette": config.get("palette") or default_palette(int(config.get("num_classes", 2))),
        "state_dict": model.state_dict(),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model,
    *,
    model_name: str,
    config: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, model_name=model_name, config=config), checkpoint_path)


def load_checkpoint_bundle(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    model_path = config.get("model_path")
    if not model_path:
        raise MLXUserError("This action requires --model-path pointing to a checkpoint.")

    checkpoint = torch.load(model_path, map_location=config.get("device", "cpu"))
    if "state_dict" not in checkpoint:
        raise MLXUserError(
            f"Checkpoint '{model_path}' does not include segmentation metadata. Re-train the model with the segmentation mode."
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
        "colored": runtime_config["colored"],
        "input_size": runtime_config["input_size"],
        "mask_threshold": float(checkpoint.get("mask_threshold", config.get("mask_threshold", 0.5))),
        "model_name": model_name,
        "num_classes": runtime_config["num_classes"],
        "palette": checkpoint.get("palette") or default_palette(runtime_config["num_classes"]),
    }
    return model, metadata


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

