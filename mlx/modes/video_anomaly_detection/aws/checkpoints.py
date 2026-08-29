from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from mlx.core.exceptions import MLXUserError


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility_fingerprint(training: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"training": dict(training), "variant": dict(variant)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class RecoveryCheckpoint:
    path: Path
    best_path: Optional[Path]
    epoch: int


class RotatingCheckpointPublisher:
    def __init__(
        self,
        *,
        recovery_dir: Path,
        output_dir: Path,
        deployment_name: str,
        fingerprint: str,
        image_uri: Optional[str],
    ) -> None:
        self.recovery_dir = recovery_dir
        self.output_dir = output_dir
        self.deployment_name = deployment_name
        self.fingerprint = fingerprint
        self.image_uri = image_uri
        self.last_epoch = 0

    def publish(self, checkpoint_path: Path) -> Optional[RecoveryCheckpoint]:
        checkpoint = _load_checkpoint(checkpoint_path)
        epoch = int(checkpoint["completed_epoch"])
        if epoch <= self.last_epoch:
            return None
        slot = self._inactive_slot()
        destination = self.recovery_dir / f"resume-{slot}.pth"
        atomic_copy(checkpoint_path, destination)
        best_source = self.output_dir / self.deployment_name
        best_destination = None
        best_digest = None
        if best_source.is_file():
            _load_deployment_checkpoint(best_source)
            best_destination = self.recovery_dir / f"best-{slot}.pth"
            atomic_copy(best_source, best_destination)
            best_digest = sha256_file(best_destination)
        atomic_json(
            self.recovery_dir / f"resume-{slot}.json",
            {
                "version": 1,
                "slot": slot,
                "epoch": epoch,
                "fingerprint": self.fingerprint,
                "image_uri": self.image_uri,
                "checkpoint_sha256": sha256_file(destination),
                "best_sha256": best_digest,
            },
        )
        atomic_json(
            self.recovery_dir / "current.json",
            {"version": 1, "slot": slot, "epoch": epoch},
        )
        self.last_epoch = epoch
        return RecoveryCheckpoint(destination, best_destination, epoch)

    def _inactive_slot(self) -> str:
        try:
            current = json.loads((self.recovery_dir / "current.json").read_text(encoding="utf-8"))
            return "b" if current.get("slot") == "a" else "a"
        except (OSError, json.JSONDecodeError):
            return "a"


def find_valid_recovery_checkpoint(
    recovery_dir: Path,
    *,
    fingerprint: str,
    image_uri: Optional[str],
) -> Optional[RecoveryCheckpoint]:
    candidates: list[RecoveryCheckpoint] = []
    saw_state = False
    for slot in ("a", "b"):
        checkpoint_path = recovery_dir / f"resume-{slot}.pth"
        metadata_path = recovery_dir / f"resume-{slot}.json"
        saw_state = saw_state or checkpoint_path.exists() or metadata_path.exists()
        if not checkpoint_path.is_file() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != fingerprint:
                continue
            if image_uri and metadata.get("image_uri") != image_uri:
                continue
            if metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path):
                continue
            checkpoint = _load_checkpoint(checkpoint_path)
            if int(checkpoint["completed_epoch"]) != int(metadata["epoch"]):
                continue
            best = recovery_dir / f"best-{slot}.pth"
            if not best.is_file() or metadata.get("best_sha256") != sha256_file(best):
                best = None
            elif best is not None:
                _load_deployment_checkpoint(best)
            candidates.append(
                RecoveryCheckpoint(checkpoint_path, best, int(metadata["epoch"]))
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, MLXUserError):
            continue
    if candidates:
        return max(candidates, key=lambda item: item.epoch)
    if saw_state:
        raise MLXUserError(f"No valid video-anomaly recovery checkpoint exists in {recovery_dir}.")
    return None


def restore_recovery(
    recovery: RecoveryCheckpoint,
    *,
    output_dir: Path,
    deployment_name: str,
) -> Path:
    last = output_dir / deployment_name.replace(".pth", ".last.pth")
    atomic_copy(recovery.path, last)
    if recovery.best_path:
        atomic_copy(recovery.best_path, output_dir / deployment_name)
    return last


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Unable to validate recovery checkpoint '{path}': {exc}") from exc
    required = {
        "mode", "training_state_version", "completed_epoch", "optimizer_state_dict",
        "history", "state_dict", "model_name",
    }
    if not isinstance(value, Mapping) or required - set(value):
        raise MLXUserError(f"Checkpoint '{path}' is not a complete video-anomaly training state.")
    epoch = int(value["completed_epoch"])
    if value.get("mode") != "video_anomaly_detection" or epoch < 1 or len(value["history"]) != epoch:
        raise MLXUserError(f"Checkpoint '{path}' has inconsistent video-anomaly state.")
    return value


def _load_deployment_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Unable to validate deployment checkpoint '{path}': {exc}") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("mode") != "video_anomaly_detection"
        or "state_dict" not in value
    ):
        raise MLXUserError(f"Checkpoint '{path}' is not a video-anomaly deployment state.")
    return value


__all__ = [
    "RecoveryCheckpoint",
    "RotatingCheckpointPublisher",
    "atomic_copy",
    "atomic_json",
    "compatibility_fingerprint",
    "find_valid_recovery_checkpoint",
    "restore_recovery",
    "sha256_file",
]
