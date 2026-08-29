from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mlx.core.commands import CallbackWorkflowReporter, WorkflowEvent
from mlx.core.datasets import extract_zip_safely
from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.artifacts import resolve_training_paths
from mlx.modes.video_anomaly_detection.aws.checkpoints import (
    RotatingCheckpointPublisher,
    atomic_json,
    compatibility_fingerprint,
    find_valid_recovery_checkpoint,
    restore_recovery,
    sha256_file,
)
from mlx.modes.video_anomaly_detection.data import video_anomaly_dataset_root
from mlx.modes.video_anomaly_detection.requests import TrainVideoAnomalyRequest
from mlx.modes.video_anomaly_detection.training import TrainVideoAnomalyModel


SAGEMAKER_HYPERPARAMETERS = Path("/opt/ml/input/config/hyperparameters.json")
SAGEMAKER_INPUT = Path("/opt/ml/input/data/training")
SAGEMAKER_CHECKPOINTS = Path("/opt/ml/checkpoints")
SAGEMAKER_MODEL = Path("/opt/ml/model")
WORK_DIR = Path("/tmp/mlx-video-anomaly")
DATASET_DIR = Path("/tmp/mlx-video-anomaly-dataset")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunSageMakerVideoAnomalyBatchTraining:
    def __init__(
        self,
        *,
        hyperparameters_path: Path = SAGEMAKER_HYPERPARAMETERS,
        input_dir: Path = SAGEMAKER_INPUT,
        checkpoint_dir: Path = SAGEMAKER_CHECKPOINTS,
        model_dir: Path = SAGEMAKER_MODEL,
        work_dir: Path = WORK_DIR,
        dataset_dir: Path = DATASET_DIR,
        trainer_factory=TrainVideoAnomalyModel,
    ) -> None:
        self.hyperparameters_path = hyperparameters_path
        self.input_dir = input_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.dataset_dir = dataset_dir
        self.trainer_factory = trainer_factory
        self._manifest: dict[str, Any] = {}
        self._manifest_path = checkpoint_dir / "batch-status.json"
        self._publisher: Optional[RotatingCheckpointPublisher] = None
        self._epochs = 0

    def execute(self) -> dict[str, Any]:
        hyperparameters = self._load_hyperparameters()
        training = self._json_parameter(hyperparameters, "mlx_training", dict)
        variants = self._json_parameter(hyperparameters, "mlx_variants", list)
        batch_id = str(hyperparameters.get("mlx_batch_id") or "").strip()
        image_uri = str(hyperparameters.get("mlx_image_uri") or "") or None
        if not batch_id or not variants:
            raise MLXUserError("SageMaker batch hyperparameters are missing the batch ID or model inventory.")
        self._validate_variants(variants)
        self._epochs = int(training.get("epochs", 50))
        dataset_root = self._extract_dataset(
            max_uncompressed_bytes=int(hyperparameters.get("mlx_volume_size_gb", 200))
            * 700_000_000
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._load_or_initialize_manifest(batch_id, variants, self._epochs)

        for index, variant in enumerate(variants):
            state = self._variant_state(str(variant["variant_id"]))
            if state["status"] == "completed":
                try:
                    self._verify_completed_variant(state)
                except MLXUserError as exc:
                    state.update(
                        {"status": "failed", "error": str(exc), "failed_at": _timestamp()}
                    )
                    self._manifest.update(
                        {
                            "status": "failed",
                            "current_variant": state["variant_id"],
                            "failure_reason": str(exc),
                            "updated_at": _timestamp(),
                        }
                    )
                    self._write_manifest()
                    raise
                continue
            self._run_variant(index, variant, training, dataset_root, image_uri)

        self._manifest.update(
            {
                "status": "completed",
                "current_variant": None,
                "failure_reason": None,
                "updated_at": _timestamp(),
            }
        )
        self._write_manifest()
        shutil.copytree(
            self.checkpoint_dir / "models",
            self.model_dir / "models",
            dirs_exist_ok=True,
        )
        atomic_json(self.model_dir / "batch-status.json", self._manifest)
        return self._manifest

    def _run_variant(
        self,
        index: int,
        variant: Mapping[str, Any],
        training: Mapping[str, Any],
        dataset_root: Path,
        image_uri: Optional[str],
    ) -> None:
        variant_id = str(variant["variant_id"])
        output_dir = self.work_dir / variant_id
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)
        config = {
            **training,
            "model": str(variant["model_name"]),
            "drax_fusion_mode": variant.get("drax_fusion_mode") or "average",
            "dataset_path": str(dataset_root),
            "output_path": str(output_dir),
            "device": self._resolve_device(),
            "backbone_mode": "3d",
            "backbone_mode_explicit": True,
        }
        paths = resolve_training_paths(config)
        fingerprint = compatibility_fingerprint(training, variant)
        recovery_dir = self.checkpoint_dir / "recovery" / variant_id
        recovery = find_valid_recovery_checkpoint(
            recovery_dir,
            fingerprint=fingerprint,
            image_uri=image_uri,
        )
        if recovery is not None:
            config["model_path"] = str(
                restore_recovery(
                    recovery,
                    output_dir=output_dir,
                    deployment_name=paths["checkpoint"].name,
                )
            )
        self._publisher = RotatingCheckpointPublisher(
            recovery_dir=recovery_dir,
            output_dir=output_dir,
            deployment_name=paths["checkpoint"].name,
            fingerprint=fingerprint,
            image_uri=image_uri,
        )
        if recovery is not None:
            self._publisher.last_epoch = recovery.epoch
        state = self._variant_state(variant_id)
        state.update(
            {
                "status": "running",
                "completed_epoch": recovery.epoch if recovery else 0,
                "error": None,
                "started_at": state.get("started_at") or _timestamp(),
            }
        )
        self._manifest.update(
            {
                "status": "running",
                "current_variant": variant_id,
                "failure_reason": None,
                "updated_at": _timestamp(),
            }
        )
        self._write_manifest()
        try:
            reporter = CallbackWorkflowReporter(
                lambda event: self._handle_event(event, index, variant_id)
            )
            self.trainer_factory(
                TrainVideoAnomalyRequest.from_config(config), reporter=reporter
            ).execute()
            self._publisher.publish(paths["last_checkpoint"])
            destination = self.checkpoint_dir / "models" / variant_id
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(output_dir, destination)
            files = {
                str(path.relative_to(destination)): sha256_file(path)
                for path in sorted(destination.rglob("*"))
                if path.is_file()
            }
            atomic_json(destination / "artifact-manifest.json", {"version": 1, "files": files})
            state.update(
                {
                    "status": "completed",
                    "completed_epoch": self._epochs,
                    "completed_at": _timestamp(),
                    "artifact_manifest": f"models/{variant_id}/artifact-manifest.json",
                }
            )
            self._manifest["current_variant"] = None
            self._manifest["updated_at"] = _timestamp()
            self._write_manifest()
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception as exc:
            state.update({"status": "failed", "error": str(exc), "failed_at": _timestamp()})
            self._manifest.update(
                {
                    "status": "failed",
                    "current_variant": variant_id,
                    "failure_reason": str(exc),
                    "updated_at": _timestamp(),
                }
            )
            self._write_manifest()
            raise
        finally:
            self._publisher = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    def _handle_event(self, event: WorkflowEvent, index: int, variant_id: str) -> None:
        payload = event.payload
        if not isinstance(payload, Mapping) or payload.get("event") != "video_anomaly_training_epoch":
            return
        checkpoint_path = payload.get("checkpoint_path")
        if checkpoint_path and self._publisher is not None:
            recovery = self._publisher.publish(Path(str(checkpoint_path)))
            if recovery is not None:
                state = self._variant_state(variant_id)
                state["completed_epoch"] = recovery.epoch
                self._manifest["updated_at"] = _timestamp()
                self._write_manifest()
                overall = ((index + recovery.epoch / max(self._epochs, 1)) / len(self._manifest["variants"])) * 100
                print(
                    f"MLX_MODEL_INDEX={index + 1}; MLX_EPOCH={recovery.epoch}; "
                    f"MLX_PROGRESS={overall:.4f};",
                    flush=True,
                )

    def _load_or_initialize_manifest(
        self,
        batch_id: str,
        variants: list[Any],
        epochs: int,
    ) -> None:
        if self._manifest_path.is_file():
            try:
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MLXUserError(f"Unable to restore batch status: {exc}") from exc
            if not isinstance(manifest, Mapping) or manifest.get("version") != 1:
                raise MLXUserError("Recovered batch status has an unsupported structure or version.")
            expected = [item["variant_id"] for item in variants]
            actual = [item.get("variant_id") for item in manifest.get("variants", [])]
            if manifest.get("batch_id") != batch_id or actual != expected:
                raise MLXUserError("Recovered batch status does not match the frozen batch inventory.")
            self._manifest = manifest
            return
        self._manifest = {
            "version": 1,
            "batch_id": batch_id,
            "status": "pending",
            "current_variant": None,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "variants": [
                {
                    **item,
                    "status": "pending",
                    "completed_epoch": 0,
                    "total_epochs": epochs,
                    "error": None,
                }
                for item in variants
            ],
        }
        self._write_manifest()

    def _variant_state(self, variant_id: str) -> dict[str, Any]:
        for state in self._manifest["variants"]:
            if state["variant_id"] == variant_id:
                return state
        raise MLXUserError(f"Batch status is missing model variant '{variant_id}'.")

    def _verify_completed_variant(self, state: Mapping[str, Any]) -> None:
        variant_id = str(state["variant_id"])
        directory = self.checkpoint_dir / "models" / variant_id
        manifest_path = directory / "artifact-manifest.json"
        try:
            artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = artifact_manifest["files"]
            if not isinstance(files, Mapping):
                raise ValueError("files")
            required = {"training.csv", "training_history.png", "run_metadata.json"}
            if not required.issubset(files):
                raise ValueError("required artifacts")
            if not any(name.endswith(".last.pth") for name in files):
                raise ValueError("resumable checkpoint")
            if not any(name.endswith(".pth") and not name.endswith(".last.pth") for name in files):
                raise ValueError("deployment checkpoint")
            for relative, digest in files.items():
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(relative)
                path = directory / relative_path
                if not path.is_file() or sha256_file(path) != digest:
                    raise ValueError(relative)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MLXUserError(
                f"Completed model '{variant_id}' has missing or corrupt S3 artifacts."
            ) from exc

    def _write_manifest(self) -> None:
        atomic_json(self._manifest_path, self._manifest)

    def _extract_dataset(self, *, max_uncompressed_bytes: int) -> Path:
        archives = sorted(path for path in self.input_dir.rglob("*.zip") if path.is_file())
        if len(archives) != 1:
            raise MLXUserError(
                f"Expected exactly one dataset ZIP in {self.input_dir}; found {len(archives)}."
            )
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True)
        extract_zip_safely(
            archives[0], self.dataset_dir, max_uncompressed_bytes=max_uncompressed_bytes
        )
        return video_anomaly_dataset_root(self.dataset_dir)

    def _load_hyperparameters(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.hyperparameters_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MLXUserError(f"Unable to read SageMaker hyperparameters: {exc}") from exc
        if not isinstance(value, Mapping):
            raise MLXUserError("SageMaker hyperparameters must be a JSON object.")
        return value

    @staticmethod
    def _json_parameter(values: Mapping[str, Any], name: str, expected_type):
        raw = values.get(name)
        if not isinstance(raw, str):
            raise MLXUserError(f"SageMaker hyperparameters are missing '{name}'.")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MLXUserError(f"Invalid SageMaker parameter '{name}': {exc}") from exc
        if not isinstance(value, expected_type):
            raise MLXUserError(f"SageMaker parameter '{name}' has the wrong type.")
        return value

    @staticmethod
    def _validate_variants(variants: list[Any]) -> None:
        identifiers: list[str] = []
        for item in variants:
            if not isinstance(item, Mapping):
                raise MLXUserError("Each frozen model variant must be an object.")
            model_name = str(item.get("model_name") or "").strip()
            variant_id = str(item.get("variant_id") or "").strip()
            if not model_name or not variant_id:
                raise MLXUserError("Each frozen model variant requires model_name and variant_id.")
            identifiers.append(variant_id)
        if len(set(identifiers)) != len(identifiers):
            raise MLXUserError("Frozen model variant IDs must be unique.")

    @staticmethod
    def _resolve_device() -> str:
        return "cuda:0" if int(os.environ.get("SM_NUM_GPUS", "0")) > 0 else "cpu"


def main() -> int:
    try:
        RunSageMakerVideoAnomalyBatchTraining().execute()
    except KeyboardInterrupt:
        return 143
    except MLXUserError as exc:
        print(f"MLX SageMaker video-anomaly training error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
