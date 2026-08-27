from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Optional

from mlx.core.commands import CallbackWorkflowReporter, WorkflowEvent
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.aws.checkpoints import (
    RotatingCheckpointPublisher,
    find_valid_recovery_checkpoint,
    prepare_working_resume_checkpoint,
)
from mlx.modes.object_detection.commands import TrainObjectDetectionModel
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest


SAGEMAKER_HYPERPARAMETERS = Path("/opt/ml/input/config/hyperparameters.json")
SAGEMAKER_INPUT = Path("/opt/ml/input/data/training")
SAGEMAKER_CHECKPOINTS = Path("/opt/ml/checkpoints")
SAGEMAKER_MODEL = Path("/opt/ml/model")
WORK_DIR = Path("/tmp/mlx-training")
DATASET_DIR = Path("/tmp/mlx-dataset")


class RunSageMakerObjectDetectionTraining:
    def __init__(
        self,
        *,
        hyperparameters_path: Path = SAGEMAKER_HYPERPARAMETERS,
        input_dir: Path = SAGEMAKER_INPUT,
        checkpoint_dir: Path = SAGEMAKER_CHECKPOINTS,
        model_dir: Path = SAGEMAKER_MODEL,
        work_dir: Path = WORK_DIR,
        dataset_dir: Path = DATASET_DIR,
    ) -> None:
        self.hyperparameters_path = hyperparameters_path
        self.input_dir = input_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.dataset_dir = dataset_dir
        self._publisher: Optional[RotatingCheckpointPublisher] = None

    def execute(self) -> Any:
        hyperparameters = self._load_hyperparameters()
        training = self._training_config(hyperparameters)
        provider = str(training.get("provider") or "ultralytics")
        total_epochs = int(training.get("epochs", 100))
        volume_size_gb = int(hyperparameters.get("mlx_volume_size_gb", 100))
        run_name = str(training.get("run_name") or f"mlx-{provider}")
        compatibility_fingerprint = self._compatibility_fingerprint(training)
        image_uri = str(hyperparameters.get("mlx_image_uri") or "") or None
        required_resume = str(hyperparameters.get("mlx_resume", "false")).lower() == "true"
        recovery = find_valid_recovery_checkpoint(
            self.checkpoint_dir,
            expected_provider=provider,
            required=required_resume,
            expected_fingerprint=compatibility_fingerprint,
            expected_image_uri=image_uri,
        )

        dataset_root = self._extract_dataset(max_uncompressed_bytes=volume_size_gb * 700_000_000)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if recovery is not None:
            prepare_working_resume_checkpoint(
                recovery,
                work_dir=self.work_dir,
                run_name=run_name,
                provider=provider,
                total_epochs=total_epochs,
            )

        training.update(
            {
                "dataset_path": str(dataset_root),
                "output_path": str(self.work_dir),
                "run_name": run_name,
                "device": self._resolve_device(str(training.get("device") or "auto")),
                "save_period": -1,
            }
        )
        request = TrainObjectDetectionRequest.from_config(training)
        self._publisher = RotatingCheckpointPublisher(
            work_dir=self.work_dir,
            recovery_dir=self.checkpoint_dir,
            provider=provider,
            total_epochs=total_epochs,
            compatibility_fingerprint=compatibility_fingerprint,
            image_uri=image_uri,
            initial_epoch=recovery.epoch if recovery is not None else 0,
            progress_emitter=self._emit_sagemaker_metrics,
        )
        self._publisher.start()
        previous_handler = signal.signal(signal.SIGTERM, self._handle_sigterm)
        try:
            reporter = CallbackWorkflowReporter(self._handle_workflow_event)
            result = TrainObjectDetectionModel(request, reporter=reporter).execute()
            self._publisher.publish_now()
            self._stage_model_artifacts(result, training)
            return result
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            self._publisher.stop()

    def _handle_sigterm(self, signum, frame) -> None:
        if self._publisher is not None:
            try:
                self._publisher.publish_now()
            except MLXUserError:
                pass
        raise KeyboardInterrupt("SageMaker requested training termination")

    def _handle_workflow_event(self, event: WorkflowEvent) -> None:
        payload = event.payload
        if (
            self._publisher is not None
            and isinstance(payload, Mapping)
            and payload.get("checkpoint_path")
        ):
            self._publisher.publish_path(Path(str(payload["checkpoint_path"])))

    @staticmethod
    def _emit_sagemaker_metrics(epoch: int, progress: float, eta: float) -> None:
        print(
            f"MLX_EPOCH={epoch}; MLX_PROGRESS={progress:.4f}; "
            f"MLX_ETA_SECONDS={eta:.2f};",
            flush=True,
        )

    def _load_hyperparameters(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.hyperparameters_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MLXUserError(
                f"Unable to read SageMaker hyperparameters from {self.hyperparameters_path}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise MLXUserError("SageMaker hyperparameters must be a JSON object.")
        return value

    @staticmethod
    def _training_config(hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
        raw = hyperparameters.get("mlx_training")
        if not isinstance(raw, str):
            raise MLXUserError("SageMaker hyperparameters are missing 'mlx_training'.")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MLXUserError(f"Invalid SageMaker MLX training configuration: {exc}") from exc
        if not isinstance(value, dict):
            raise MLXUserError("The SageMaker MLX training configuration must be an object.")
        return value

    def _extract_dataset(self, *, max_uncompressed_bytes: Optional[int] = None) -> Path:
        archives = sorted(path for path in self.input_dir.rglob("*.zip") if path.is_file())
        if len(archives) != 1:
            raise MLXUserError(
                f"Expected exactly one dataset ZIP in {self.input_dir}; found {len(archives)}."
            )
        archive = archives[0]
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True)
        try:
            with zipfile.ZipFile(archive) as source:
                total_size = sum(member.file_size for member in source.infolist())
                if max_uncompressed_bytes is not None and total_size > max_uncompressed_bytes:
                    raise MLXUserError(
                        "The extracted dataset would exceed the safe portion of the configured "
                        "SageMaker volume. Increase aws.volume_size_gb."
                    )
                for member in source.infolist():
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise MLXUserError(
                            f"Dataset ZIP contains an unsafe path: {member.filename}"
                        )
                    mode = member.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise MLXUserError(
                            f"Dataset ZIP contains a symbolic link: {member.filename}"
                        )
                source.extractall(self.dataset_dir)
        except zipfile.BadZipFile as exc:
            raise MLXUserError(f"Dataset ZIP is invalid: {archive}") from exc
        data_files = sorted(self.dataset_dir.rglob("data.yaml"))
        if len(data_files) != 1:
            raise MLXUserError(
                "The extracted dataset must contain exactly one data.yaml; "
                f"found {len(data_files)}."
            )
        return data_files[0].parent

    @staticmethod
    def _resolve_device(value: str) -> str:
        if value != "auto":
            return value
        return "0" if int(os.environ.get("SM_NUM_GPUS", "0")) > 0 else "cpu"

    @staticmethod
    def _compatibility_fingerprint(training: Mapping[str, Any]) -> str:
        immutable = {
            key: value
            for key, value in training.items()
            if key not in {"device", "epochs", "run_name", "save_period"}
        }
        payload = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _stage_model_artifacts(self, result: Any, training: Mapping[str, Any]) -> None:
        candidate = None
        if isinstance(result, Mapping):
            candidate = result.get("checkpoint_path") or result.get("model_path")
        else:
            candidate = getattr(result, "checkpoint_path", None) or getattr(
                result, "model_path", None
            )
        if candidate and Path(candidate).is_file():
            shutil.copy2(candidate, self.model_dir / Path(candidate).name)
        else:
            checkpoints = [
                path
                for pattern in ("best.pt", "last.pt")
                for path in self.work_dir.rglob(pattern)
                if path.is_file()
            ]
            if checkpoints:
                selected = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
                shutil.copy2(selected, self.model_dir / selected.name)
        benchmark_result = getattr(result, "benchmark_result", None)
        benchmark_output = getattr(benchmark_result, "output_dir", None)
        if benchmark_output and Path(benchmark_output).is_dir():
            shutil.copytree(
                benchmark_output,
                self.model_dir / "benchmark",
                dirs_exist_ok=True,
            )
        (self.model_dir / "training-summary.json").write_text(
            json.dumps(dict(training), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def main() -> int:
    try:
        RunSageMakerObjectDetectionTraining().execute()
    except KeyboardInterrupt:
        return 143
    except MLXUserError as exc:
        print(f"MLX SageMaker training error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
