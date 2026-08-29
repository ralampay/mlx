from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True)
class RecoveryCheckpoint:
    checkpoint_path: Path
    best_checkpoint_path: Optional[Path]
    epoch: int
    model_name: str
    family: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path, *, resumable: bool) -> dict[str, Any]:
    try:
        import torch
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise MLXUserError(f"Unable to validate image-classification checkpoint '{path}': {exc}") from exc
    if not isinstance(value, dict) or "state_dict" not in value:
        raise MLXUserError(f"Image-classification checkpoint '{path}' has no model state.")
    if resumable:
        required = {
            "training_state_version",
            "completed_epoch",
            "optimizer_state_dict",
            "history",
            "model_name",
            "family",
        }
        missing = sorted(required - set(value))
        if missing or value.get("training_state_version") != 1:
            raise MLXUserError(f"Checkpoint '{path}' is not a complete resumable image-classification checkpoint.")
        epoch = int(value["completed_epoch"])
        if epoch < 1 or len(value["history"]) != epoch:
            raise MLXUserError(f"Checkpoint '{path}' has inconsistent completed-epoch history.")
    return value


class RotatingCheckpointPublisher:
    def __init__(
        self,
        *,
        recovery_dir: Path,
        output_dir: Path,
        model_name: str,
        compatibility_fingerprint: str,
        image_uri: Optional[str],
        total_epochs: int,
        initial_epoch: int = 0,
        progress_emitter: Optional[
            Callable[[int, float, float], None]
        ] = None,
    ) -> None:
        self.recovery_dir = recovery_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.compatibility_fingerprint = compatibility_fingerprint
        self.image_uri = image_uri
        self.total_epochs = total_epochs
        self.progress_emitter, self._last_epoch = progress_emitter, initial_epoch
        self._initial_epoch = initial_epoch
        self._started_at = time.monotonic()

    def publish_path(self, source: Path) -> Optional[RecoveryCheckpoint]:
        checkpoint = _load_checkpoint(source, resumable=True)
        epoch = int(checkpoint["completed_epoch"])
        if epoch <= self._last_epoch:
            return None
        slot = self._inactive_slot()
        destination = self.recovery_dir / f"resume-{slot}.pth"
        _atomic_copy(source, destination)
        best_source = self.output_dir / f"{self.model_name}.pth"
        best_digest = None
        if best_source.is_file():
            _load_checkpoint(best_source, resumable=False)
            slot_best = self.recovery_dir / f"best-{slot}.pth"
            _atomic_copy(best_source, slot_best)
            _atomic_copy(best_source, self.recovery_dir / "best.pth")
            best_digest = _sha256(slot_best)
        metadata = {
            "version": 1, "slot": slot, "epoch": epoch,
            "model_name": checkpoint["model_name"], "family": checkpoint["family"],
            "compatibility_fingerprint": self.compatibility_fingerprint, "image_uri": self.image_uri,
            "checkpoint_sha256": _sha256(destination), "best_sha256": best_digest,
        }
        _atomic_json(self.recovery_dir / f"resume-{slot}.json", metadata)
        _atomic_json(self.recovery_dir / "current.json", {"version": 1, "slot": slot, "epoch": epoch})
        self._last_epoch = epoch
        elapsed = time.monotonic() - self._started_at
        completed_this_attempt = max(epoch - self._initial_epoch, 1)
        eta = max(0.0, (self.total_epochs - epoch) * elapsed / completed_this_attempt)
        if self.progress_emitter:
            self.progress_emitter(
                epoch,
                min(100.0, epoch / max(self.total_epochs, 1) * 100),
                eta,
            )
        return RecoveryCheckpoint(
            destination,
            self.recovery_dir / f"best-{slot}.pth" if best_digest else None,
            epoch,
            checkpoint["model_name"],
            checkpoint["family"],
        )

    def _inactive_slot(self) -> str:
        current = self.recovery_dir / "current.json"
        try:
            return "b" if json.loads(current.read_text(encoding="utf-8")).get("slot") == "a" else "a"
        except (OSError, json.JSONDecodeError):
            return "a"


def find_valid_recovery_checkpoint(
    recovery_dir: Path,
    *,
    required: bool,
    expected_fingerprint: Optional[str] = None,
    expected_image_uri: Optional[str] = None,
) -> Optional[RecoveryCheckpoint]:
    candidates: list[RecoveryCheckpoint] = []
    saw_files = False
    for slot in ("a", "b"):
        checkpoint_path = recovery_dir / f"resume-{slot}.pth"
        metadata_path = recovery_dir / f"resume-{slot}.json"
        saw_files = saw_files or checkpoint_path.exists() or metadata_path.exists()
        if not checkpoint_path.is_file() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                expected_fingerprint is not None
                and metadata.get("compatibility_fingerprint") != expected_fingerprint
            ):
                continue
            if expected_image_uri is not None and metadata.get("image_uri") != expected_image_uri:
                continue
            if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
                continue
            checkpoint = _load_checkpoint(checkpoint_path, resumable=True)
            epoch = int(checkpoint["completed_epoch"])
            if epoch != int(metadata["epoch"]):
                continue
            best_path = recovery_dir / f"best-{slot}.pth"
            if not best_path.is_file() or metadata.get("best_sha256") != _sha256(
                best_path
            ):
                best_path = None
            elif best_path is not None:
                _load_checkpoint(best_path, resumable=False)
            candidates.append(
                RecoveryCheckpoint(
                    checkpoint_path,
                    best_path,
                    epoch,
                    str(checkpoint["model_name"]),
                    str(checkpoint["family"]),
                )
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, MLXUserError):
            continue
    if candidates:
        return max(candidates, key=lambda item: item.epoch)
    if required or saw_files:
        raise MLXUserError(
            "No valid full-state image-classification recovery checkpoint was "
            f"restored in {recovery_dir}."
        )
    return None


def prepare_working_resume_checkpoint(
    recovery: RecoveryCheckpoint,
    *,
    output_dir: Path,
) -> Path:
    last = output_dir / f"{recovery.model_name}.last.pth"
    _atomic_copy(recovery.checkpoint_path, last)
    if recovery.best_checkpoint_path:
        _atomic_copy(recovery.best_checkpoint_path, output_dir / f"{recovery.model_name}.pth")
    return last
