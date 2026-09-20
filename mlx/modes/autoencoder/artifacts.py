from __future__ import annotations

import csv
import json
import platform
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mlx.core.artifacts import atomic_torch_save, sha256_file, write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.models import AutoencoderRegistry, DEFAULT_AUTOENCODER_REGISTRY


CHECKPOINT_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise MLXUserError(f"Autoencoder output must be a new or empty directory: {output}")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MLXUserError(f"Unable to create autoencoder output directory '{output}': {exc}") from exc
    return output


def checkpoint_payload(
    *,
    model,
    architecture_name: str,
    architecture_path: str,
    model_config: Mapping[str, Any],
    loss_name: str,
    loss_path: str,
    loss_config: Mapping[str, Any],
    expects_l2_normalized_input: bool,
    best_epoch: int,
    best_validation_loss: float,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "adapter_type": "autoencoder",
        "architecture": architecture_name,
        "architecture_path": architecture_path,
        "model_config": dict(model_config),
        "input_dimensions": int(model.input_dimensions),
        "bottleneck_dimensions": int(model.bottleneck_dimensions),
        "expects_l2_normalized_input": bool(expects_l2_normalized_input),
        "loss": loss_name,
        "loss_path": loss_path,
        "loss_config": dict(loss_config),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "source_csv": source_path.name,
        "source_sha256": sha256_file(source_path),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }


def load_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise MLXUserError(f"Autoencoder checkpoint not found: {checkpoint_path}")
    try:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise MLXUserError(f"Unable to load autoencoder checkpoint '{checkpoint_path}': {exc}") from exc
    if not isinstance(value, dict):
        raise MLXUserError(f"Autoencoder checkpoint must contain a mapping: {checkpoint_path}")
    required = {
        "checkpoint_version", "adapter_type", "architecture", "architecture_path",
        "model_config", "input_dimensions", "bottleneck_dimensions", "state_dict",
        "expects_l2_normalized_input",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise MLXUserError(
            f"Autoencoder checkpoint '{checkpoint_path}' is missing: {', '.join(missing)}."
        )
    if type(value["checkpoint_version"]) is not int or value["checkpoint_version"] != CHECKPOINT_VERSION or value["adapter_type"] != "autoencoder":
        raise MLXUserError(f"Unsupported autoencoder checkpoint schema: {checkpoint_path}")
    _validate_checkpoint(value)
    return checkpoint_path, value


def _validate_checkpoint(value: Mapping[str, Any]) -> None:
    for field in ("input_dimensions", "bottleneck_dimensions"):
        if type(value.get(field)) is not int or value[field] < 1:
            raise MLXUserError(f"Autoencoder checkpoint has invalid {field}.")
    for field in ("architecture", "architecture_path"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise MLXUserError(f"Autoencoder checkpoint requires {field}.")
    for field in ("model_config", "state_dict"):
        if not isinstance(value.get(field), Mapping):
            raise MLXUserError(f"Autoencoder checkpoint requires a {field} mapping.")
    if type(value.get("expects_l2_normalized_input")) is not bool:
        raise MLXUserError("Autoencoder checkpoint input normalization must be Boolean.")


def build_checkpoint_model(
    checkpoint: Mapping[str, Any],
    *,
    registry: AutoencoderRegistry = DEFAULT_AUTOENCODER_REGISTRY,
    device: str = "cpu",
    trust_checkpoint_code: bool = False,
):
    _validate_checkpoint(checkpoint)
    reference = checkpoint["architecture_path"]
    # Registry references are explicitly supplied by the application, not the checkpoint.
    trusted = set(DEFAULT_AUTOENCODER_REGISTRY.entries.values()) | set(registry.entries.values())
    trusted.add("mlx.modes.autoencoder.architectures.simple:SimpleAutoencoderDefinition")
    if reference not in trusted and not trust_checkpoint_code:
        raise MLXUserError(
            "Checkpoint references external architecture code. Register its exact import "
            "reference explicitly or use --trust-checkpoint-code only for a trusted checkpoint."
        )
    definition, _ = registry.resolve(str(checkpoint["architecture_path"]))
    config = dict(checkpoint["model_config"])
    config["input_dimensions"] = int(checkpoint["input_dimensions"])
    config["bottleneck_dimensions"] = int(checkpoint["bottleneck_dimensions"])
    try:
        model = definition.build(config)
    except MLXUserError:
        raise
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise MLXUserError(
            f"Unable to rebuild autoencoder architecture '{checkpoint['architecture']}': {exc}"
        ) from exc
    try:
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise MLXUserError(f"Unable to restore autoencoder model state: {exc}") from exc
    if (
        int(getattr(model, "input_dimensions", -1)) != int(checkpoint["input_dimensions"])
        or int(getattr(model, "bottleneck_dimensions", -1))
        != int(checkpoint["bottleneck_dimensions"])
        or not callable(getattr(model, "encode", None))
    ):
        raise MLXUserError("Loaded autoencoder does not satisfy its checkpoint dimension contract.")
    return model


def write_transformed_csv(path: Path, table, vectors: Sequence[Sequence[float]]) -> None:
    if path.exists():
        raise MLXUserError(f"Autoencoder embedding output already exists: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=table.fieldnames)
            writer.writeheader()
            for row, vector in zip(table.rows, vectors, strict=True):
                writer.writerow({**row, "embedding": json.dumps(list(vector), separators=(",", ":"))})
    except OSError as exc:
        raise MLXUserError(f"Unable to write autoencoder embedding CSV '{path}': {exc}") from exc


def write_training_artifacts(
    output_dir: Path,
    *,
    checkpoint: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    plots: bool,
) -> None:
    atomic_torch_save(checkpoint, output_dir / "autoencoder.pth")
    write_csv(output_dir / "training.csv", history)
    write_json_atomic(
        output_dir / "autoencoder_manifest.json",
        {key: value for key, value in checkpoint.items() if key != "state_dict"},
    )
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    if plots:
        _write_plot(output_dir / "training_history.png", history)


def _write_plot(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
    axis.set(xlabel="Epoch", ylabel="Reconstruction loss", title="Autoencoder Training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_metadata(*, action: str, started_at: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "autoencoder",
        "action": action,
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        **values,
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "build_checkpoint_model",
    "checkpoint_payload",
    "load_checkpoint",
    "prepare_output_directory",
    "run_metadata",
    "utc_timestamp",
    "write_training_artifacts",
    "write_transformed_csv",
]
