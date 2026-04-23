from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models import (
    DEFAULT_MODEL,
    build_image_classification_model,
    model_family_for,
)


def resolve_model_name(config: dict[str, Any]) -> str:
    return config.get("model") or DEFAULT_MODEL


def checkpoint_payload(
    model,
    *,
    model_name: str,
    family: str,
    config: dict[str, Any],
    classes: list[str] | None = None,
) -> dict[str, Any]:
    classes = classes or []
    return {
        "classes": classes,
        "colored": bool(config.get("colored", True)),
        "embedding_size": int(config.get("embedding_size", 4096)),
        "family": family,
        "input_size": tuple(config.get("input_size", (224, 224))),
        "model_config": dict(config),
        "model_name": model_name,
        "num_classes": len(classes) if classes else None,
        "state_dict": model.state_dict(),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model,
    *,
    model_name: str,
    family: str,
    config: dict[str, Any],
    classes: list[str] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model,
            model_name=model_name,
            family=family,
            config=config,
            classes=classes,
        ),
        checkpoint_path,
    )


def resolve_train_output_paths(config: dict[str, Any], *, model_name: str) -> dict[str, Path]:
    output_path = config.get("output_path")
    if not output_path:
        raise MLXUserError("Training requires --output pointing to the directory where artifacts will be written.")
    output_dir = Path(output_path).expanduser()
    return {
        "output_dir": output_dir,
        "checkpoint_path": output_dir / f"{model_name}.pth",
        "training_csv_path": output_dir / "training.csv",
    }


def load_checkpoint_bundle(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    model_path = config.get("model_path")
    if not model_path:
        raise MLXUserError("This action requires --model-path pointing to a checkpoint.")

    checkpoint = torch.load(model_path, map_location=config.get("device", "cpu"))
    if "state_dict" not in checkpoint:
        raise MLXUserError(
            f"Checkpoint '{model_path}' does not include metadata. Re-train the model with the new image-classification mode."
        )

    model_name = config.get("model") or checkpoint.get("model_name") or DEFAULT_MODEL
    family = model_family_for(model_name)
    checkpoint_family = checkpoint.get("family")
    if checkpoint_family and checkpoint_family != family:
        raise MLXUserError(
            f"Checkpoint family '{checkpoint_family}' does not match requested model '{model_name}'."
        )

    runtime_config = dict(checkpoint.get("model_config") or {})
    runtime_config.update(config)
    runtime_config["colored"] = checkpoint.get("colored", runtime_config.get("colored", True))
    runtime_config["embedding_size"] = checkpoint.get(
        "embedding_size", runtime_config.get("embedding_size", 4096)
    )
    runtime_config["input_size"] = tuple(
        checkpoint.get("input_size", runtime_config.get("input_size", (224, 224)))
    )
    runtime_config["pretrained"] = False

    num_classes = checkpoint.get("num_classes")
    if family == "standard" and not num_classes:
        classes = checkpoint.get("classes") or []
        num_classes = len(classes)
    model = build_image_classification_model(
        model_name,
        runtime_config,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["state_dict"])

    metadata = {
        "checkpoint_path": Path(model_path),
        "classes": checkpoint.get("classes") or [],
        "family": family,
        "input_size": runtime_config["input_size"],
        "model_name": model_name,
        "num_classes": num_classes,
        "colored": runtime_config["colored"],
    }
    return model, metadata
