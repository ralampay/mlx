from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True)
class RecoveryCheckpoint:
    checkpoint_path: Path
    state_path: Optional[Path]
    epoch: int
    provider: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    os.replace(temporary, destination)


def _atomic_json(destination: Path, value: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _load_provider_checkpoint(
    path: Path,
    *,
    provider: Optional[str] = None,
    require_full_state: bool = False,
) -> dict[str, Any]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, ValueError, EOFError) as exc:
        raise MLXUserError(f"Recovery checkpoint is unreadable: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("epoch"), int):
        raise MLXUserError(f"Recovery checkpoint has no valid completed epoch: {path}")
    if require_full_state:
        missing: list[str] = []
        if checkpoint.get("optimizer") is None:
            missing.append("optimizer")
        if provider == "ultralytics":
            for key in ("ema", "scaler", "train_args"):
                if checkpoint.get(key) is None:
                    missing.append(key)
        elif provider == "libreyolo":
            if checkpoint.get("model") is None and checkpoint.get("train_model") is None:
                missing.append("model/train_model")
            if checkpoint.get("config") is None:
                missing.append("config")
        if missing:
            raise MLXUserError(
                f"Recovery checkpoint is weights-only or incomplete ({', '.join(missing)}): {path}"
            )
    return checkpoint


def _capture_rng_state(path: Path) -> None:
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def restore_rng_state(path: Optional[Path]) -> None:
    if path is None or not path.is_file():
        return
    try:
        import numpy as np
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except (ImportError, KeyError, OSError, RuntimeError, ValueError, EOFError) as exc:
        raise MLXUserError(f"Unable to restore recovery RNG state from {path}: {exc}") from exc


class RotatingCheckpointPublisher:
    """Publish validated provider checkpoints into two atomic recovery slots."""

    def __init__(
        self,
        *,
        work_dir: Path,
        recovery_dir: Path,
        provider: str,
        total_epochs: int,
        compatibility_fingerprint: Optional[str] = None,
        image_uri: Optional[str] = None,
        initial_epoch: int = 0,
        poll_interval: float = 2.0,
        progress_emitter: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        self.work_dir = work_dir
        self.recovery_dir = recovery_dir
        self.provider = provider
        self.total_epochs = total_epochs
        self.compatibility_fingerprint = compatibility_fingerprint
        self.image_uri = image_uri
        self.poll_interval = poll_interval
        self.progress_emitter = progress_emitter
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_source_signature: Optional[tuple[int, int]] = None
        self._best_source_signature: Optional[tuple[int, int]] = None
        self._last_epoch = initial_epoch
        self._publish_lock = threading.Lock()
        self._started_at = time.monotonic()
        self._published_at: list[tuple[int, float]] = []
        self.last_error: Optional[Exception] = None

    def start(self) -> None:
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="mlx-checkpoint-publisher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval * 2))

    def publish_now(self) -> Optional[RecoveryCheckpoint]:
        source = self._find_last_checkpoint()
        if source is None:
            return None
        return self.publish_path(source)

    def publish_path(self, source: Path) -> Optional[RecoveryCheckpoint]:
        with self._publish_lock:
            return self._publish_path_locked(source)

    def _publish_path_locked(self, source: Path) -> Optional[RecoveryCheckpoint]:
        signature = (source.stat().st_mtime_ns, source.stat().st_size)
        if signature == self._last_source_signature:
            return None
        checkpoint = _load_provider_checkpoint(
            source,
            provider=self.provider,
            require_full_state=True,
        )
        epoch = int(checkpoint["epoch"]) + 1
        if epoch <= self._last_epoch:
            self._last_source_signature = signature
            return None
        slot = self._inactive_slot()
        checkpoint_path = self.recovery_dir / f"resume-{slot}.pt"
        state_path = self.recovery_dir / f"resume-{slot}.state.pt"
        metadata_path = self.recovery_dir / f"resume-{slot}.json"
        _atomic_copy(source, checkpoint_path)
        _capture_rng_state(state_path)
        metadata = {
            "version": 1,
            "slot": slot,
            "epoch": epoch,
            "provider": self.provider,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "image_uri": self.image_uri,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "state_sha256": _sha256(state_path) if state_path.exists() else None,
        }
        _atomic_json(metadata_path, metadata)
        _atomic_json(
            self.recovery_dir / "current.json",
            {"version": 1, "slot": slot, "epoch": epoch},
        )
        self._publish_best_checkpoint()
        self._last_source_signature = signature
        self._last_epoch = epoch
        now = time.monotonic()
        self._published_at.append((epoch, now))
        self._published_at = self._published_at[-10:]
        eta = self._eta_seconds(epoch, now)
        progress = min(100.0, epoch / max(self.total_epochs, 1) * 100.0)
        if self.progress_emitter is not None:
            self.progress_emitter(epoch, progress, eta)
        return RecoveryCheckpoint(checkpoint_path, state_path, epoch, self.provider)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self.publish_now()
            except MLXUserError as exc:
                self.last_error = exc

    def _find_last_checkpoint(self) -> Optional[Path]:
        candidates = [path for path in self.work_dir.rglob("last.pt") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))

    def _publish_best_checkpoint(self) -> None:
        candidates = [path for path in self.work_dir.rglob("best.pt") if path.is_file()]
        if not candidates:
            return
        source = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        signature = (source.stat().st_mtime_ns, source.stat().st_size)
        if signature == self._best_source_signature:
            return
        _load_provider_checkpoint(source)
        destination = self.recovery_dir / "best.pt"
        _atomic_copy(source, destination)
        _atomic_json(
            self.recovery_dir / "best.json",
            {
                "version": 1,
                "provider": self.provider,
                "checkpoint_sha256": _sha256(destination),
            },
        )
        self._best_source_signature = signature

    def _inactive_slot(self) -> str:
        current_path = self.recovery_dir / "current.json"
        if not current_path.is_file():
            return "a"
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "a"
        return "b" if current.get("slot") == "a" else "a"

    def _eta_seconds(self, epoch: int, now: float) -> float:
        if len(self._published_at) >= 2:
            first_epoch, first_time = self._published_at[0]
            completed = max(epoch - first_epoch, 1)
            seconds_per_epoch = (now - first_time) / completed
        else:
            seconds_per_epoch = (now - self._started_at) / max(epoch, 1)
        return max(0.0, (self.total_epochs - epoch) * seconds_per_epoch)


def find_valid_recovery_checkpoint(
    recovery_dir: Path,
    *,
    expected_provider: str,
    required: bool,
    expected_fingerprint: Optional[str] = None,
    expected_image_uri: Optional[str] = None,
) -> Optional[RecoveryCheckpoint]:
    candidates: list[RecoveryCheckpoint] = []
    saw_recovery_files = False
    for slot in ("a", "b"):
        checkpoint_path = recovery_dir / f"resume-{slot}.pt"
        state_path = recovery_dir / f"resume-{slot}.state.pt"
        metadata_path = recovery_dir / f"resume-{slot}.json"
        saw_recovery_files = saw_recovery_files or any(
            path.exists() for path in (checkpoint_path, state_path, metadata_path)
        )
        if not checkpoint_path.is_file() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("provider") != expected_provider:
                continue
            if (
                expected_fingerprint is not None
                and metadata.get("compatibility_fingerprint") != expected_fingerprint
            ):
                continue
            if expected_image_uri is not None and metadata.get("image_uri") != expected_image_uri:
                continue
            if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
                continue
            if state_path.is_file() and metadata.get("state_sha256") != _sha256(state_path):
                continue
            checkpoint = _load_provider_checkpoint(
                checkpoint_path,
                provider=expected_provider,
                require_full_state=True,
            )
            epoch = int(checkpoint["epoch"]) + 1
            if epoch != int(metadata["epoch"]):
                continue
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, MLXUserError):
            continue
        candidates.append(
            RecoveryCheckpoint(
                checkpoint_path=checkpoint_path,
                state_path=state_path if state_path.is_file() else None,
                epoch=epoch,
                provider=expected_provider,
            )
        )
    if candidates:
        return max(candidates, key=lambda item: item.epoch)
    if required or saw_recovery_files:
        raise MLXUserError(
            f"No valid full-state {expected_provider} recovery checkpoint was restored in {recovery_dir}."
        )
    return None


def prepare_working_resume_checkpoint(
    recovery: RecoveryCheckpoint,
    *,
    work_dir: Path,
    run_name: str,
    provider: str,
    total_epochs: int,
) -> Path:
    destination = work_dir / run_name / "weights" / "last.pt"
    _atomic_copy(recovery.checkpoint_path, destination)
    if provider == "ultralytics":
        checkpoint = _load_provider_checkpoint(
            destination,
            provider=provider,
            require_full_state=True,
        )
        train_args = checkpoint.get("train_args")
        if not isinstance(train_args, dict):
            raise MLXUserError("Ultralytics recovery checkpoint has no training arguments.")
        original_target = int(train_args.get("epochs", total_epochs))
        if total_epochs < original_target:
            raise MLXUserError(
                "The resumed Ultralytics epoch target cannot be lower than its original target."
            )
        if total_epochs != original_target:
            train_args["epochs"] = total_epochs
            try:
                import torch

                temporary = destination.with_name(".last.pt.tmp")
                torch.save(checkpoint, temporary)
                os.replace(temporary, destination)
            except (ImportError, OSError, RuntimeError) as exc:
                raise MLXUserError(
                    f"Unable to prepare the Ultralytics resume checkpoint: {exc}"
                ) from exc
    restore_rng_state(recovery.state_path)
    return destination
