from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from mlx.core.commands import CallbackWorkflowReporter, WorkflowEvent
from mlx.core.datasets import classification_dataset_root, extract_zip_safely
from mlx.core.exceptions import MLXUserError
from mlx.core.random import apply_global_seed
from mlx.modes.image_classification.aws.checkpoints import (
    RotatingCheckpointPublisher,
    find_valid_recovery_checkpoint,
    prepare_working_resume_checkpoint,
)
from mlx.modes.image_classification.requests import ImageClassificationRequest
from mlx.modes.image_classification.train import TrainImageClassificationModel
from mlx.modes.image_classification.utils import resolve_model_name


SAGEMAKER_HYPERPARAMETERS = Path("/opt/ml/input/config/hyperparameters.json")
SAGEMAKER_INPUT = Path("/opt/ml/input/data/training")
SAGEMAKER_CHECKPOINTS = Path("/opt/ml/checkpoints")
SAGEMAKER_MODEL = Path("/opt/ml/model")
WORK_DIR = Path("/tmp/mlx-training")
DATASET_DIR = Path("/tmp/mlx-dataset")


class RunSageMakerImageClassificationTraining:
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
        self._latest_checkpoint: Optional[Path] = None

    def execute(self) -> Any:
        hp = self._load_hyperparameters()
        training = self._training_config(hp)
        total_epochs = int(training.get("epochs", 100))
        fingerprint = self._compatibility_fingerprint(training)
        image_uri = str(hp.get("mlx_image_uri") or "") or None
        required_resume = str(hp.get("mlx_resume", "false")).lower() == "true"
        recovery = find_valid_recovery_checkpoint(
            self.checkpoint_dir,
            required=required_resume,
            expected_fingerprint=fingerprint,
            expected_image_uri=image_uri,
        )
        dataset_root = self._extract_dataset(
            max_uncompressed_bytes=int(hp.get("mlx_volume_size_gb", 100))
            * 700_000_000
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        output_dir = self.work_dir / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = resolve_model_name(training)
        if recovery:
            if recovery.model_name != model_name:
                raise MLXUserError(f"Recovery model '{recovery.model_name}' does not match '{model_name}'.")
            training["model_path"] = str(
                prepare_working_resume_checkpoint(recovery, output_dir=output_dir)
            )
        training.update(
            {
                "dataset_path": str(dataset_root),
                "output_path": str(output_dir),
                "device": self._resolve_device(
                    str(training.get("device") or "auto")
                ),
            }
        )
        apply_global_seed(training.get("random_seed"))
        self._publisher = RotatingCheckpointPublisher(
            recovery_dir=self.checkpoint_dir, output_dir=output_dir, model_name=model_name,
            compatibility_fingerprint=fingerprint, image_uri=image_uri, total_epochs=total_epochs,
            initial_epoch=recovery.epoch if recovery else 0, progress_emitter=self._emit_metrics,
        )
        previous = signal.signal(signal.SIGTERM, self._handle_sigterm)
        try:
            result = TrainImageClassificationModel(
                ImageClassificationRequest.from_config(training),
                reporter=CallbackWorkflowReporter(self._handle_event),
            ).execute()
            self._stage_artifacts(output_dir, training)
            return result
        finally:
            signal.signal(signal.SIGTERM, previous)

    def _handle_event(self, event: WorkflowEvent) -> None:
        if (
            isinstance(event.payload, Mapping)
            and event.payload.get("checkpoint_path")
            and self._publisher
        ):
            self._latest_checkpoint = Path(str(event.payload["checkpoint_path"]))
            self._publisher.publish_path(self._latest_checkpoint)

    def _handle_sigterm(self, signum, frame) -> None:
        if self._publisher and self._latest_checkpoint and self._latest_checkpoint.is_file():
            try:
                self._publisher.publish_path(self._latest_checkpoint)
            except MLXUserError:
                pass
        raise KeyboardInterrupt("SageMaker requested training termination")

    @staticmethod
    def _emit_metrics(epoch: int, progress: float, eta: float) -> None:
        print(f"MLX_EPOCH={epoch}; MLX_PROGRESS={progress:.4f}; MLX_ETA_SECONDS={eta:.2f};", flush=True)

    def _load_hyperparameters(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.hyperparameters_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MLXUserError(
                f"Unable to read SageMaker hyperparameters from "
                f"{self.hyperparameters_path}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise MLXUserError("SageMaker hyperparameters must be a JSON object.")
        return value

    @staticmethod
    def _training_config(hp: Mapping[str, Any]) -> dict[str, Any]:
        raw = hp.get("mlx_training")
        if not isinstance(raw, str):
            raise MLXUserError(
                "SageMaker hyperparameters are missing 'mlx_training'."
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MLXUserError(
                f"Invalid SageMaker MLX training configuration: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise MLXUserError(
                "The SageMaker MLX training configuration must be an object."
            )
        return value

    def _extract_dataset(self, *, max_uncompressed_bytes: int) -> Path:
        archives = sorted(path for path in self.input_dir.rglob("*.zip") if path.is_file())
        if len(archives) != 1:
            raise MLXUserError(
                f"Expected exactly one dataset ZIP in {self.input_dir}; "
                f"found {len(archives)}."
            )
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True)
        extract_zip_safely(
            archives[0],
            self.dataset_dir,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
        return classification_dataset_root(self.dataset_dir)

    @staticmethod
    def _resolve_device(value: str) -> str:
        if value != "auto":
            return value
        return "cuda:0" if int(os.environ.get("SM_NUM_GPUS", "0")) > 0 else "cpu"

    @staticmethod
    def _compatibility_fingerprint(training: Mapping[str, Any]) -> str:
        immutable = {
            key: value
            for key, value in training.items()
            if key not in {"device", "epochs"}
        }
        payload = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _stage_artifacts(self, output_dir: Path, training: Mapping[str, Any]) -> None:
        names = (
            f"{resolve_model_name(training)}.pth",
            f"{resolve_model_name(training)}.last.pth",
            "training.csv",
        )
        missing = [name for name in names if not (output_dir / name).is_file()]
        if missing:
            raise MLXUserError(
                "Image-classification training completed without required artifact(s): "
                + ", ".join(missing)
            )
        try:
            for name in names:
                shutil.copy2(output_dir / name, self.model_dir / name)
            summary = {
                key: value
                for key, value in training.items()
                if key not in {"dataset_path", "output_path", "model_path"}
            }
            (self.model_dir / "training-summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise MLXUserError(f"Unable to stage SageMaker model artifacts: {exc}") from exc


def main() -> int:
    try:
        RunSageMakerImageClassificationTraining().execute()
    except KeyboardInterrupt:
        return 143
    except MLXUserError as exc:
        print(f"MLX SageMaker image-classification training error: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(
            f"MLX SageMaker image-classification training failed unexpectedly: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
