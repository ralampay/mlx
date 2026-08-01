from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models import (
    DEFAULT_MODEL,
    build_image_classification_model,
    model_family_for,
)
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier


def _ood_metadata(model, config: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(model, JointDeepSVDDClassifier):
        return None
    threshold = model.svdd_threshold.detach().cpu()
    return {
        "method": "deep-svdd",
        "center": model.svdd_center.detach().cpu(),
        "threshold": float(threshold.item()) if torch.isfinite(threshold) else None,
        "quantile": float(config.get("svdd_quantile", 0.95)),
        "embedding_dim": int(model.svdd_center.numel()),
        "hidden_dim": int(model.svdd_head[0].out_features),
        "score_type": "squared_euclidean",
    }


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
    model_config = dict(config)
    if config.get("ood_method", "none") == "none":
        for key in (
            "ood_method",
            "svdd_weight",
            "svdd_dim",
            "svdd_hidden_dim",
            "svdd_quantile",
            "svdd_warmup_epochs",
        ):
            model_config.pop(key, None)
    payload = {
        "classes": classes,
        "colored": bool(config.get("colored", True)),
        "embedding_size": int(config.get("embedding_size", 4096)),
        "family": family,
        "input_size": tuple(config.get("input_size", (224, 224))),
        "model_config": model_config,
        "model_name": model_name,
        "num_classes": len(classes) if classes else None,
        "state_dict": model.state_dict(),
    }
    ood = _ood_metadata(model, config)
    if ood is not None:
        payload["ood"] = ood
    return payload


def save_checkpoint(
    checkpoint_path: Path,
    model,
    *,
    model_name: str,
    family: str,
    config: dict[str, Any],
    classes: list[str] | None = None,
) -> None:
    _atomic_torch_save(
        checkpoint_payload(
            model,
            model_name=model_name,
            family=family,
            config=config,
            classes=classes,
        ),
        checkpoint_path,
    )


def save_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    *,
    model_name: str,
    family: str,
    config: dict[str, Any],
    completed_epoch: int,
    best_val_loss: float,
    history: list[dict[str, float | int]],
    classes: list[str] | None = None,
) -> None:
    payload = checkpoint_payload(
        model,
        model_name=model_name,
        family=family,
        config=config,
        classes=classes,
    )
    payload.update(
        {
            "training_state_version": 1,
            "completed_epoch": int(completed_epoch),
            "best_val_loss": float(best_val_loss),
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
    family: str,
    config: dict[str, Any],
    classes: list[str] | None = None,
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
        raise MLXUserError(f"Could not load resume checkpoint '{path}': {exc}") from exc

    if checkpoint.get("training_state_version") != 1:
        raise MLXUserError(
            f"Checkpoint '{path}' is not a resumable image-classification training checkpoint."
        )
    if checkpoint.get("model_name") != model_name:
        raise MLXUserError(
            f"Resume checkpoint model '{checkpoint.get('model_name')}' does not match '{model_name}'."
        )
    if checkpoint.get("family") != family:
        raise MLXUserError(
            f"Resume checkpoint family '{checkpoint.get('family')}' does not match '{family}'."
        )

    expected_classes = classes or []
    if list(checkpoint.get("classes") or []) != expected_classes:
        raise MLXUserError(
            "Resume checkpoint class labels do not match the selected dataset."
        )
    expected_input_size = tuple(config.get("input_size", (224, 224)))
    if tuple(checkpoint.get("input_size") or ()) != expected_input_size:
        raise MLXUserError(
            "Resume checkpoint input size does not match the current --width/--height settings."
        )
    if bool(checkpoint.get("colored", True)) != bool(config.get("colored", True)):
        raise MLXUserError(
            "Resume checkpoint color mode does not match the current training configuration."
        )

    checkpoint_config = checkpoint.get("model_config") or {}
    if checkpoint_config.get("drax_fusion_mode", "average") != config.get(
        "drax_fusion_mode", "average"
    ):
        raise MLXUserError(
            "Resume checkpoint Drax fusion mode does not match the current configuration."
        )

    checkpoint_ood = checkpoint.get("ood") or {"method": "none"}
    requested_ood = config.get("ood_method", "none")
    if checkpoint_ood.get("method", "none") != requested_ood:
        raise MLXUserError(
            "Resume checkpoint OOD method does not match the current --ood-method setting. "
            "Use the same OOD configuration that created the checkpoint."
        )
    if requested_ood == "deep-svdd":
        expected = (
            int(config.get("svdd_dim", 128)),
            int(config.get("svdd_hidden_dim", 256)),
        )
        actual = (
            int(checkpoint_ood.get("embedding_dim", -1)),
            int(checkpoint_ood.get("hidden_dim", -1)),
        )
        if actual != expected:
            raise MLXUserError(
                "Resume checkpoint Deep SVDD dimensions do not match --svdd-dim and --svdd-hidden-dim."
            )

    try:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise MLXUserError(
            f"Resume checkpoint '{path}' is incompatible with the current model: {exc}"
        ) from exc

    _restore_random_state(checkpoint.get("random_state") or {})
    history = checkpoint.get("history") or []
    completed_epoch = int(checkpoint.get("completed_epoch", 0))
    if len(history) != completed_epoch:
        raise MLXUserError(
            f"Resume checkpoint '{path}' has inconsistent epoch history."
        )

    return {
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "completed_epoch": completed_epoch,
        "history": history,
    }


def resolve_train_output_paths(config: dict[str, Any], *, model_name: str) -> dict[str, Path]:
    output_path = config.get("output_path")
    if not output_path:
        raise MLXUserError("Training requires --output pointing to the directory where artifacts will be written.")
    output_dir = Path(output_path).expanduser()
    return {
        "output_dir": output_dir,
        "checkpoint_path": output_dir / f"{model_name}.pth",
        "last_checkpoint_path": output_dir / f"{model_name}.last.pth",
        "training_csv_path": output_dir / "training.csv",
    }


def load_checkpoint_bundle(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    model_path = config.get("model_path")
    if not model_path:
        raise MLXUserError("This action requires --model-path pointing to a checkpoint.")

    checkpoint = torch.load(
        model_path,
        map_location=config.get("device", "cpu"),
        weights_only=True,
    )
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
    checkpoint_ood = checkpoint.get("ood") or {"method": "none"}
    runtime_config["ood_method"] = checkpoint_ood.get("method", "none")
    if runtime_config["ood_method"] == "deep-svdd":
        runtime_config["svdd_dim"] = int(checkpoint_ood.get("embedding_dim", 128))
        runtime_config["svdd_hidden_dim"] = int(checkpoint_ood.get("hidden_dim", 256))
        runtime_config["svdd_quantile"] = float(checkpoint_ood.get("quantile", 0.95))
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
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        if runtime_config["ood_method"] == "deep-svdd":
            raise MLXUserError(
                f"Checkpoint '{model_path}' has incompatible or incomplete Deep SVDD state: {exc}"
            ) from exc
        raise MLXUserError(f"Checkpoint '{model_path}' is incompatible with '{model_name}': {exc}") from exc
    if isinstance(model, JointDeepSVDDClassifier):
        metadata_threshold = checkpoint_ood.get("threshold")
        if metadata_threshold is None:
            model.svdd_threshold.fill_(float("nan"))
        else:
            model.svdd_threshold.fill_(float(metadata_threshold))

    metadata = {
        "checkpoint_path": Path(model_path),
        "classes": checkpoint.get("classes") or [],
        "family": family,
        "input_size": runtime_config["input_size"],
        "model_name": model_name,
        "num_classes": num_classes,
        "colored": runtime_config["colored"],
        "ood": checkpoint_ood,
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
        temporary_path.unlink(missing_ok=True)


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
