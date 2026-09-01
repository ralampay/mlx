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
from mlx.modes.image_recognition_oc.artifacts import resolve_training_paths
from mlx.modes.image_recognition_oc.aws.checkpoints import (
    RotatingCheckpointPublisher,
    atomic_json,
    compatibility_fingerprint,
    find_valid_recovery_checkpoint,
    restore_recovery,
    sha256_file,
)
from mlx.modes.image_recognition_oc.data import image_one_class_dataset_root
from mlx.modes.image_recognition_oc.evaluation import BenchmarkImageOneClass
from mlx.modes.image_recognition_oc.requests import (
    BenchmarkImageOneClassRequest,
    TrainImageOneClassRequest,
)
from mlx.modes.image_recognition_oc.training import TrainImageOneClassModel


SAGEMAKER_HYPERPARAMETERS = Path("/opt/ml/input/config/hyperparameters.json")
SAGEMAKER_INPUT = Path("/opt/ml/input/data/training")
SAGEMAKER_MODEL_INPUT = Path("/opt/ml/input/data/model")
SAGEMAKER_CHECKPOINTS = Path("/opt/ml/checkpoints")
SAGEMAKER_MODEL = Path("/opt/ml/model")
WORK_DIR = Path("/tmp/mlx-image-oc")
DATASET_DIR = Path("/tmp/mlx-image-oc-dataset")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunSageMakerImageOneClass:
    def __init__(
        self,
        *,
        hyperparameters_path: Path = SAGEMAKER_HYPERPARAMETERS,
        input_dir: Path = SAGEMAKER_INPUT,
        model_input_dir: Path = SAGEMAKER_MODEL_INPUT,
        checkpoint_dir: Path = SAGEMAKER_CHECKPOINTS,
        model_dir: Path = SAGEMAKER_MODEL,
        work_dir: Path = WORK_DIR,
        dataset_dir: Path = DATASET_DIR,
        trainer_factory=TrainImageOneClassModel,
        benchmark_factory=BenchmarkImageOneClass,
    ) -> None:
        self.hyperparameters_path = hyperparameters_path
        self.input_dir = input_dir
        self.model_input_dir = model_input_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.dataset_dir = dataset_dir
        self.trainer_factory = trainer_factory
        self.benchmark_factory = benchmark_factory
        self._manifest: dict[str, Any] = {}
        self._manifest_path = checkpoint_dir / "run-status.json"
        self._publisher: Optional[RotatingCheckpointPublisher] = None
        self._active_paths: Optional[dict[str, Path]] = None
        self._epochs = 0

    def execute(self) -> dict[str, Any]:
        hp = self._load_hyperparameters()
        operation = str(hp.get("mlx_operation") or "").strip()
        training = self._json_parameter(hp, "mlx_training", dict)
        benchmark = self._json_parameter(hp, "mlx_benchmark", dict)
        variants = self._json_parameter(hp, "mlx_variants", list)
        run_id = str(hp.get("mlx_run_id") or "").strip()
        image_uri = str(hp.get("mlx_image_uri") or "") or None
        if operation not in {"train", "train-all", "benchmark"} or not run_id:
            raise MLXUserError("SageMaker hyperparameters are missing a valid operation or run ID.")
        self._validate_variants(variants, allow_empty=operation == "benchmark")
        dataset_root = self._extract_dataset(
            max_uncompressed_bytes=int(hp.get("mlx_volume_size_gb", 100)) * 700_000_000
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if operation == "benchmark":
            return self._run_standalone_benchmark(run_id, benchmark, dataset_root)
        self._epochs = int(training.get("epochs", 50))
        self._load_or_initialize_manifest(run_id, operation, variants, self._epochs, bool(benchmark.get("enabled", False)))
        for index, variant in enumerate(variants):
            self._run_or_resume_variant(index, variant, training, benchmark, dataset_root, image_uri)
        self._manifest.update(
            {"status": "completed", "current_variant": None, "failure_reason": None, "updated_at": _timestamp()}
        )
        self._write_manifest()
        shutil.copytree(self.checkpoint_dir / "models", self.model_dir / "models", dirs_exist_ok=True)
        atomic_json(self.model_dir / "run-status.json", self._manifest)
        return self._manifest

    def _run_or_resume_variant(
        self,
        index: int,
        variant: Mapping[str, Any],
        training: Mapping[str, Any],
        benchmark: Mapping[str, Any],
        dataset_root: Path,
        image_uri: Optional[str],
    ) -> None:
        variant_id = str(variant["variant_id"])
        state = self._variant_state(variant_id)
        benchmark_enabled = bool(benchmark.get("enabled", False))
        if state["status"] == "completed":
            self._verify_training_artifacts(state)
            if not benchmark_enabled or state["benchmark_status"] == "completed":
                if benchmark_enabled:
                    self._verify_benchmark_artifacts(state)
                else:
                    self._verify_final_training_artifacts(state)
                return
            self._benchmark_variant(state, benchmark, dataset_root)
            return

        output_dir = self.work_dir / variant_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        config = {
            **training,
            "model": str(variant["model_name"]),
            "backbone": str(variant["backbone_name"]),
            "drax_fusion_mode": variant.get("drax_fusion_mode") or "average",
            "dataset_path": str(dataset_root),
            "output_path": str(output_dir),
            "device": self._resolve_device(),
        }
        paths = resolve_training_paths(config)
        fingerprint = compatibility_fingerprint(training, variant)
        recovery_dir = self.checkpoint_dir / "recovery" / variant_id
        recovery = find_valid_recovery_checkpoint(recovery_dir, fingerprint=fingerprint, image_uri=image_uri)
        if recovery is not None:
            config["model_path"] = str(
                restore_recovery(recovery, output_dir=output_dir, deployment_name=paths["checkpoint"].name)
            )
        self._publisher = RotatingCheckpointPublisher(
            recovery_dir=recovery_dir,
            output_dir=output_dir,
            deployment_name=paths["checkpoint"].name,
            fingerprint=fingerprint,
            image_uri=image_uri,
        )
        self._active_paths = paths
        if recovery is not None:
            self._publisher.last_epoch = recovery.epoch
        state.update(
            {
                "status": "running",
                "benchmark_status": "pending" if benchmark_enabled else "not_requested",
                "completed_epoch": recovery.epoch if recovery else 0,
                "error": None,
                "started_at": state.get("started_at") or _timestamp(),
            }
        )
        self._manifest.update(
            {"status": "running", "current_variant": variant_id, "failure_reason": None, "updated_at": _timestamp()}
        )
        self._write_manifest()
        try:
            reporter = CallbackWorkflowReporter(
                lambda event: self._handle_event(event, index, variant_id)
            )
            self.trainer_factory(
                TrainImageOneClassRequest.from_config(config), reporter=reporter
            ).execute()
            self._publisher.publish(paths["last_checkpoint"])
            destination = self.checkpoint_dir / "models" / variant_id
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(output_dir, destination)
            self._write_artifact_manifest(destination, "training-artifact-manifest.json")
            if not benchmark_enabled:
                self._write_artifact_manifest(destination, "artifact-manifest.json")
            state.update(
                {
                    "status": "completed",
                    "completed_epoch": self._epochs,
                    "completed_at": _timestamp(),
                    "training_artifact_manifest": f"models/{variant_id}/training-artifact-manifest.json",
                }
            )
            self._manifest["current_variant"] = None
            self._manifest["updated_at"] = _timestamp()
            self._write_manifest()
            if benchmark_enabled:
                self._benchmark_variant(state, benchmark, dataset_root)
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception as exc:
            if state.get("status") != "completed":
                state["status"] = "failed"
            state.update({"error": str(exc), "failed_at": _timestamp()})
            self._manifest.update(
                {"status": "failed", "current_variant": variant_id, "failure_reason": str(exc), "updated_at": _timestamp()}
            )
            self._write_manifest()
            raise
        finally:
            self._publisher = None
            self._active_paths = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    def _benchmark_variant(
        self,
        state: dict[str, Any],
        benchmark: Mapping[str, Any],
        dataset_root: Path,
    ) -> None:
        variant_id = str(state["variant_id"])
        destination = self.checkpoint_dir / "models" / variant_id
        checkpoint = self._deployment_checkpoint(destination)
        state.update({"benchmark_status": "running", "error": None})
        self._manifest.update({"status": "running", "current_variant": variant_id, "updated_at": _timestamp()})
        self._write_manifest()
        try:
            self.benchmark_factory(
                BenchmarkImageOneClassRequest.from_config(
                    {
                        "model": None,
                        "backbone": None,
                        "model_path": str(checkpoint),
                        "dataset_path": str(dataset_root),
                        "output_path": str(destination / "benchmark"),
                        "device": self._resolve_device(),
                        "batch_size": int(benchmark.get("batch_size", 16)),
                        "workers": int(benchmark.get("workers", 0)),
                        "plots": bool(benchmark.get("plots", True)),
                    }
                )
            ).execute()
            self._write_artifact_manifest(destination, "artifact-manifest.json")
            state.update({"benchmark_status": "completed", "benchmark_completed_at": _timestamp(), "error": None})
            self._manifest.update({"current_variant": None, "failure_reason": None, "updated_at": _timestamp()})
            self._write_manifest()
        except Exception as exc:
            state.update({"benchmark_status": "failed", "error": str(exc), "failed_at": _timestamp()})
            self._manifest.update(
                {"status": "failed", "current_variant": variant_id, "failure_reason": str(exc), "updated_at": _timestamp()}
            )
            self._write_manifest()
            raise

    def _run_standalone_benchmark(
        self, run_id: str, benchmark: Mapping[str, Any], dataset_root: Path
    ) -> dict[str, Any]:
        checkpoints = sorted(path for path in self.model_input_dir.rglob("*.pth") if path.is_file())
        if len(checkpoints) != 1:
            raise MLXUserError(
                f"Expected exactly one .pth checkpoint in {self.model_input_dir}; found {len(checkpoints)}."
            )
        output = self.checkpoint_dir / "artifacts"
        manifest = {
            "version": 1,
            "run_id": run_id,
            "operation": "benchmark",
            "status": "running",
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "variants": [],
        }
        self._manifest = manifest
        self._write_manifest()
        try:
            result = self.benchmark_factory(
                BenchmarkImageOneClassRequest.from_config(
                    {
                        "model": None,
                        "backbone": None,
                        "model_path": str(checkpoints[0]),
                        "dataset_path": str(dataset_root),
                        "output_path": str(output),
                        "device": self._resolve_device(),
                        "batch_size": int(benchmark.get("batch_size", 16)),
                        "workers": int(benchmark.get("workers", 0)),
                        "plots": bool(benchmark.get("plots", True)),
                    }
                )
            ).execute()
            self._write_artifact_manifest(output, "artifact-manifest.json")
            manifest.update({"status": "completed", "updated_at": _timestamp()})
            self._write_manifest()
            shutil.copytree(output, self.model_dir / "artifacts", dirs_exist_ok=True)
            atomic_json(self.model_dir / "run-status.json", manifest)
            return {"manifest": manifest, "metrics": result["metrics"]}
        except Exception as exc:
            manifest.update({"status": "failed", "failure_reason": str(exc), "updated_at": _timestamp()})
            self._write_manifest()
            raise

    def _handle_event(self, event: WorkflowEvent, index: int, variant_id: str) -> None:
        payload = event.payload
        if not isinstance(payload, Mapping) or payload.get("event") != "image_one_class_training_epoch":
            return
        if self._publisher is None or self._active_paths is None:
            return
        recovery = self._publisher.publish(self._active_paths["last_checkpoint"])
        if recovery is None:
            return
        state = self._variant_state(variant_id)
        state["completed_epoch"] = recovery.epoch
        self._manifest["updated_at"] = _timestamp()
        self._write_manifest()
        overall = ((index + recovery.epoch / max(self._epochs, 1)) / len(self._manifest["variants"])) * 100
        print(
            f"MLX_MODEL_INDEX={index + 1}; MLX_EPOCH={recovery.epoch}; MLX_PROGRESS={overall:.4f};",
            flush=True,
        )

    def _load_or_initialize_manifest(
        self,
        run_id: str,
        operation: str,
        variants: list[Any],
        epochs: int,
        benchmark_enabled: bool,
    ) -> None:
        if self._manifest_path.is_file():
            try:
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MLXUserError(f"Unable to restore one-class run status: {exc}") from exc
            expected = [item["variant_id"] for item in variants]
            actual = [item.get("variant_id") for item in manifest.get("variants", [])]
            if manifest.get("version") != 1 or manifest.get("run_id") != run_id or manifest.get("operation") != operation or actual != expected:
                raise MLXUserError("Recovered run status does not match the frozen SVDD inventory.")
            self._manifest = manifest
            return
        self._manifest = {
            "version": 1,
            "run_id": run_id,
            "operation": operation,
            "status": "pending",
            "current_variant": None,
            "failure_reason": None,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "variants": [
                {
                    **item,
                    "status": "pending",
                    "benchmark_status": "pending" if benchmark_enabled else "not_requested",
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
        raise MLXUserError(f"Run status is missing SVDD variant '{variant_id}'.")

    def _verify_training_artifacts(self, state: Mapping[str, Any]) -> None:
        directory = self.checkpoint_dir / "models" / str(state["variant_id"])
        self._verify_artifact_manifest(
            directory,
            "training-artifact-manifest.json",
            required={"training.csv", "training_history.png", "run_metadata.json"},
            require_checkpoints=True,
        )

    def _verify_benchmark_artifacts(self, state: Mapping[str, Any]) -> None:
        directory = self.checkpoint_dir / "models" / str(state["variant_id"])
        self._verify_artifact_manifest(
            directory,
            "artifact-manifest.json",
            required={"benchmark/metrics.json", "benchmark/predictions.csv", "benchmark/run_metadata.json"},
            require_checkpoints=True,
        )

    def _verify_final_training_artifacts(self, state: Mapping[str, Any]) -> None:
        directory = self.checkpoint_dir / "models" / str(state["variant_id"])
        self._verify_artifact_manifest(
            directory,
            "artifact-manifest.json",
            required={"training.csv", "training_history.png", "run_metadata.json"},
            require_checkpoints=True,
        )

    @staticmethod
    def _verify_artifact_manifest(
        directory: Path,
        manifest_name: str,
        *,
        required: set[str],
        require_checkpoints: bool,
    ) -> None:
        try:
            manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
            files = manifest["files"]
            if not isinstance(files, Mapping) or not required.issubset(files):
                raise ValueError("required artifacts")
            if require_checkpoints:
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
                f"Completed variant '{directory.name}' has missing or corrupt S3 artifacts."
            ) from exc

    @staticmethod
    def _write_artifact_manifest(directory: Path, name: str) -> None:
        files = {
            str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in {"artifact-manifest.json", "training-artifact-manifest.json"}
        }
        atomic_json(directory / name, {"version": 1, "files": files})

    @staticmethod
    def _deployment_checkpoint(directory: Path) -> Path:
        checkpoints = sorted(
            path for path in directory.glob("*.pth") if not path.name.endswith(".last.pth")
        )
        if len(checkpoints) != 1:
            raise MLXUserError(
                f"Expected one deployment checkpoint for '{directory.name}'; found {len(checkpoints)}."
            )
        return checkpoints[0]

    def _write_manifest(self) -> None:
        atomic_json(self._manifest_path, self._manifest)

    def _extract_dataset(self, *, max_uncompressed_bytes: int) -> Path:
        archives = sorted(path for path in self.input_dir.rglob("*.zip") if path.is_file())
        if len(archives) != 1:
            raise MLXUserError(f"Expected exactly one dataset ZIP in {self.input_dir}; found {len(archives)}.")
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True)
        extract_zip_safely(archives[0], self.dataset_dir, max_uncompressed_bytes=max_uncompressed_bytes)
        return image_one_class_dataset_root(self.dataset_dir)

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
    def _validate_variants(variants: list[Any], *, allow_empty: bool) -> None:
        if not variants and not allow_empty:
            raise MLXUserError("SageMaker training requires a frozen SVDD variant inventory.")
        identifiers = []
        for item in variants:
            if not isinstance(item, Mapping):
                raise MLXUserError("Each frozen SVDD variant must be an object.")
            required = ("variant_id", "model_name", "backbone_name")
            if any(not str(item.get(key) or "").strip() for key in required):
                raise MLXUserError("Each frozen SVDD variant requires variant_id, model_name, and backbone_name.")
            if item["model_name"] != "deep-svdd":
                raise MLXUserError("The frozen AWS inventory supports only deep-svdd.")
            identifiers.append(str(item["variant_id"]))
        if len(set(identifiers)) != len(identifiers):
            raise MLXUserError("Frozen SVDD variant IDs must be unique.")

    @staticmethod
    def _resolve_device() -> str:
        return "cuda:0" if int(os.environ.get("SM_NUM_GPUS", "0")) > 0 else "cpu"


def main() -> int:
    try:
        RunSageMakerImageOneClass().execute()
    except KeyboardInterrupt:
        return 143
    except MLXUserError as exc:
        print(f"MLX SageMaker one-class image error: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"MLX SageMaker one-class image failed unexpectedly: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
