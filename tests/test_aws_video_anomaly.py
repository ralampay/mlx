from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from mlx.core.aws.models import AwsInfrastructure
from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.artifacts import resolve_training_paths
from mlx.modes.video_anomaly_detection.aws.config import (
    AwsVideoAnomalyTrainingConfig,
    load_aws_training_config,
)
from mlx.modes.video_anomaly_detection.aws.entrypoint import (
    RunSageMakerVideoAnomalyBatchTraining,
)
from mlx.modes.video_anomaly_detection.aws.service import (
    SageMakerVideoAnomalyBatchService,
)
from mlx.modes.video_anomaly_detection.variants import video_anomaly_model_variants


def _config(path: Path) -> None:
    path.write_text(
        """version: 1
aws:
  region: ap-southeast-1
  dataset_s3_uri: s3://mlx-video-anomaly-datasets/avenue-video-anomaly-detection.zip
  output_s3_uri: s3://results/video-anomaly/avenue
  instance_type: ml.g6e.2xlarge
training:
  epochs: 5
  batch_size: 1
  pretrained: true
""",
        encoding="utf-8",
    )


def test_batch_config_uses_video_defaults_and_rejects_model(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _config(path)

    loaded = load_aws_training_config(
        str(path),
        {"epochs": 7, "batch_size": 8, "_explicit_options": {"epochs"}},
    )

    assert loaded.resource_prefix == "mlx-vad"
    assert loaded.volume_size_gb == 200
    assert loaded.managed_spot is True
    assert loaded.training["epochs"] == 7
    assert loaded.training["batch_size"] == 1
    assert loaded.training["device"] == "auto"
    assert loaded.training["backbone_mode"] == "3d"

    with pytest.raises(MLXUserError, match="complete 3D model inventory"):
        load_aws_training_config(
            str(path), {"model": "resnet18", "_explicit_options": {"model"}}
        )
    with pytest.raises(MLXUserError, match="architecture option"):
        load_aws_training_config(
            str(path),
            {"temporal_model": "tcn", "_explicit_options": {"temporal_model"}},
        )


def test_documented_example_config_is_valid() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/video_anomaly_detection/aws-training.example.yaml"
    )

    loaded = load_aws_training_config(str(path), {"_explicit_options": set()})

    assert loaded.instance_type == "ml.g6e.2xlarge"
    assert loaded.dataset_s3_uri.endswith("avenue-video-anomaly-detection.zip")
    assert loaded.training["batch_size"] == 1


def test_live_batch_inventory_has_every_3d_drax_variant() -> None:
    variants = video_anomaly_model_variants()

    assert len(variants) == len({item.model_name for item in variants}) + 2
    assert len({item.variant_id for item in variants}) == len(variants)
    assert {item.variant_id for item in variants if item.model_name == "draxnet"} == {
        "draxnet-average",
        "draxnet-sknet",
    }
    assert not any(item.model_name.startswith("siamese-") for item in variants)


def test_submit_creates_one_sequential_sagemaker_job() -> None:
    class FakeS3:
        def __init__(self):
            self.objects = []

        def put_object(self, **kwargs):
            self.objects.append(kwargs)

    class FakeSageMaker:
        request = None

        def create_training_job(self, **kwargs):
            self.request = kwargs
            return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/job"}

    config = AwsVideoAnomalyTrainingConfig(
        dataset_s3_uri="s3://datasets/avenue.zip",
        output_s3_uri="s3://results/avenue",
        instance_type="ml.g6e.2xlarge",
        training={"epochs": 5, "batch_size": 1},
    )
    service = object.__new__(SageMakerVideoAnomalyBatchService)
    service.config = config
    service.region = "us-east-1"
    service.s3 = FakeS3()
    service.sagemaker = FakeSageMaker()
    service._client_error = RuntimeError
    service._boto_error = OSError

    result = service.submit(
        AwsInfrastructure(
            "us-east-1", "123", "arn:aws:iam::123:role/mlx", "image@sha256:abc"
        ),
        batch_id="a" * 32,
    )

    request = service.sagemaker.request
    expected_variants = video_anomaly_model_variants()
    assert result.model_count == len(expected_variants)
    assert request["ResourceConfig"]["InstanceCount"] == 1
    assert request["EnableManagedSpotTraining"] is True
    assert request["CheckpointConfig"]["S3Uri"].endswith(
        f"/mlx-vad/batches/{'a' * 32}"
    )
    assert len(json.loads(request["HyperParameters"]["mlx_variants"])) == len(
        expected_variants
    )
    assert len(service.s3.objects) == 2


def _dataset_zip(input_dir: Path) -> None:
    input_dir.mkdir()
    with zipfile.ZipFile(input_dir / "avenue.zip", "w") as archive:
        archive.writestr("avenue/train/normal/clip-1/001.jpg", b"image")
        archive.writestr("avenue/val/normal/clip-2/001.jpg", b"image")


def _hyperparameters(path: Path, variants: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "mlx_batch_id": "batch-1",
                "mlx_training": json.dumps(
                    {
                        "epochs": 2,
                        "batch_size": 1,
                        "lr": 0.001,
                        "clip_length": 1,
                        "backbone_temporal_kernel_size": 1,
                    }
                ),
                "mlx_variants": json.dumps(variants),
                "mlx_volume_size_gb": "1",
                "mlx_image_uri": "image@sha256:abc",
            }
        ),
        encoding="utf-8",
    )


def test_entrypoint_trains_sequentially_and_resume_skips_completed(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _dataset_zip(input_dir)
    hyperparameters = tmp_path / "hyperparameters.json"
    variants = [
        {"model_name": "resnet18", "variant_id": "resnet18", "drax_fusion_mode": None},
        {"model_name": "draxnet", "variant_id": "draxnet-sknet", "drax_fusion_mode": "sknet"},
    ]
    _hyperparameters(hyperparameters, variants)
    calls: list[tuple[str, str]] = []

    class FakeTrainer:
        def __init__(self, request, *, reporter):
            self.request = request

        def execute(self):
            config = self.request.to_config()
            calls.append((config["model"], config["drax_fusion_mode"]))
            paths = resolve_training_paths(config)
            paths["output_dir"].mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "mode": "video_anomaly_detection",
                    "training_state_version": 1,
                    "completed_epoch": 2,
                    "optimizer_state_dict": {},
                    "history": [{"epoch": 1}, {"epoch": 2}],
                    "state_dict": {},
                    "model_name": config["model"],
                },
                paths["last_checkpoint"],
            )
            torch.save(
                {
                    "mode": "video_anomaly_detection",
                    "state_dict": {},
                    "model_name": config["model"],
                },
                paths["checkpoint"],
            )
            paths["training_csv"].write_text("epoch\n1\n2\n", encoding="utf-8")
            paths["training_plot"].write_bytes(b"plot")
            paths["run_metadata"].write_text("{}", encoding="utf-8")
            return {"checkpoint_path": str(paths["checkpoint"])}

    checkpoints = tmp_path / "checkpoints"
    command = RunSageMakerVideoAnomalyBatchTraining(
        hyperparameters_path=hyperparameters,
        input_dir=input_dir,
        checkpoint_dir=checkpoints,
        model_dir=tmp_path / "model",
        work_dir=tmp_path / "work",
        dataset_dir=tmp_path / "dataset",
        trainer_factory=FakeTrainer,
    )
    result = command.execute()

    assert calls == [("resnet18", "average"), ("draxnet", "sknet")]
    assert result["status"] == "completed"
    assert (checkpoints / "models/resnet18/artifact-manifest.json").is_file()
    assert (checkpoints / "models/draxnet-sknet/artifact-manifest.json").is_file()

    second = RunSageMakerVideoAnomalyBatchTraining(
        hyperparameters_path=hyperparameters,
        input_dir=input_dir,
        checkpoint_dir=checkpoints,
        model_dir=tmp_path / "model-resume",
        work_dir=tmp_path / "work-resume",
        dataset_dir=tmp_path / "dataset-resume",
        trainer_factory=FakeTrainer,
    )
    second.execute()
    assert len(calls) == 2

    (checkpoints / "models/resnet18/training.csv").write_text(
        "corrupt", encoding="utf-8"
    )
    with pytest.raises(MLXUserError, match="missing or corrupt"):
        second.execute()
    failed = json.loads(
        (checkpoints / "batch-status.json").read_text(encoding="utf-8")
    )
    assert failed["status"] == "failed"
    assert failed["current_variant"] == "resnet18"


def test_entrypoint_fails_fast_and_records_variant(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _dataset_zip(input_dir)
    hyperparameters = tmp_path / "hyperparameters.json"
    variants = [
        {"model_name": "resnet18", "variant_id": "resnet18", "drax_fusion_mode": None},
        {"model_name": "resnet50", "variant_id": "resnet50", "drax_fusion_mode": None},
    ]
    _hyperparameters(hyperparameters, variants)

    class FailingTrainer:
        def __init__(self, request, *, reporter):
            self.request = request

        def execute(self):
            raise MLXUserError(f"cannot train {self.request.model}")

    checkpoints = tmp_path / "checkpoints"
    with pytest.raises(MLXUserError, match="cannot train resnet18"):
        RunSageMakerVideoAnomalyBatchTraining(
            hyperparameters_path=hyperparameters,
            input_dir=input_dir,
            checkpoint_dir=checkpoints,
            model_dir=tmp_path / "model",
            work_dir=tmp_path / "work",
            dataset_dir=tmp_path / "dataset",
            trainer_factory=FailingTrainer,
        ).execute()

    manifest = json.loads((checkpoints / "batch-status.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["current_variant"] == "resnet18"
    assert manifest["variants"][0]["status"] == "failed"
    assert manifest["variants"][1]["status"] == "pending"


def test_status_combines_sagemaker_and_batch_manifest() -> None:
    manifest = {
        "current_variant": "resnet50",
        "variants": [
            {"variant_id": "resnet18", "model_name": "resnet18", "drax_fusion_mode": None, "status": "completed", "completed_epoch": 5, "total_epochs": 5},
            {"variant_id": "resnet50", "model_name": "resnet50", "drax_fusion_mode": None, "status": "running", "completed_epoch": 2, "total_epochs": 5},
        ],
    }

    class Body:
        def read(self):
            return json.dumps(manifest).encode()

    class FakeS3:
        def get_object(self, **_kwargs):
            return {"Body": Body()}

    class FakeSageMaker:
        def describe_training_job(self, **_kwargs):
            return {
                "TrainingJobName": "job",
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/job",
                "TrainingJobStatus": "InProgress",
                "HyperParameters": {"mlx_batch_id": "batch"},
                "CheckpointConfig": {"S3Uri": "s3://results/batch"},
                "OutputDataConfig": {"S3OutputPath": "s3://results/output"},
            }

    service = object.__new__(SageMakerVideoAnomalyBatchService)
    service.region = "us-east-1"
    service.s3 = FakeS3()
    service.sagemaker = FakeSageMaker()
    service._client_error = RuntimeError
    service._boto_error = OSError

    status = service.status("job")

    assert status.current_variant == "resnet50"
    assert status.completed_models == 1
    assert status.total_models == 2
    assert status.progress_percent == pytest.approx(70.0)


def test_runner_routes_aws_without_changing_local_actions(monkeypatch) -> None:
    from mlx.modes.video_anomaly_detection import runner
    from mlx.modes.video_anomaly_detection.aws import runner as aws_runner

    monkeypatch.setattr(
        aws_runner,
        "run_aws_video_anomaly_detection",
        lambda config: (config["platform"], config["action"]),
    )

    assert runner.run_video_anomaly_detection(
        {"platform": "aws", "action": "train-all"}
    ) == ("aws", "train-all")
