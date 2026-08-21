from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection import runner as detection_runner
from mlx.modes.object_detection.aws.checkpoints import (
    RotatingCheckpointPublisher,
    find_valid_recovery_checkpoint,
    prepare_working_resume_checkpoint,
)
from mlx.modes.object_detection.aws.commands import WatchAwsObjectDetectionTraining
from mlx.modes.object_detection.aws.config import (
    AwsTrainingConfig,
    AwsVpcConfig,
    load_aws_training_config,
)
from mlx.modes.object_detection.aws.entrypoint import RunSageMakerObjectDetectionTraining
from mlx.modes.object_detection.aws.models import (
    AwsInfrastructure,
    AwsTrainingStatus,
)
from mlx.modes.object_detection.aws.service import SageMakerTrainingService
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest
from mlx.modes.object_detection.ultralytics.provider import UltralyticsProvider


def _write_config(path: Path, *, checkpoint_uri: str = "s3://checkpoints") -> None:
    path.write_text(
        f"""
version: 1
aws:
  dataset_s3_uri: s3://datasets/training.zip
  checkpoint_s3_uri: {checkpoint_uri}
  instance_type: ml.g4dn.xlarge
training:
  model: yolo26
  epochs: 12
""",
        encoding="utf-8",
    )


def test_aws_config_defaults_to_spot_and_accepts_bucket_root(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _write_config(path)

    config = load_aws_training_config(str(path), {"_explicit_options": set()})

    assert config.managed_spot is True
    assert config.effective_max_wait_seconds == 172800
    assert config.checkpoint_s3_uri == "s3://checkpoints"
    assert config.training.device == "auto"
    assert config.training.epochs == 12


def test_local_detection_training_does_not_read_aws_config(monkeypatch) -> None:
    calls = []

    class FakeCommand:
        def __init__(self, request, *, reporter=None):
            calls.append(request)

        def execute(self):
            return "local"

    monkeypatch.setattr(detection_runner, "TrainObjectDetectionModel", FakeCommand)

    result = detection_runner.run_object_detection(
        {
            "platform": "local",
            "action": "train",
            "config_path": "/does/not/exist.yaml",
            "model": "yolo26",
        }
    )

    assert result == "local"
    assert len(calls) == 1


def test_only_explicit_cli_training_values_override_yaml(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _write_config(path)

    config = load_aws_training_config(
        str(path),
        {
            "epochs": 20,
            "batch_size": 1,
            "_explicit_options": {"epochs"},
        },
    )

    assert config.training.epochs == 20
    assert config.training.batch_size == 16


def test_explicit_cli_profile_overrides_yaml_profile(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _write_config(path)
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("aws:\n", "aws:\n  profile: yaml-profile\n"), encoding="utf-8")

    yaml_config = load_aws_training_config(
        str(path),
        {"profile": None, "_explicit_options": set()},
    )
    cli_config = load_aws_training_config(
        str(path),
        {"profile": "cli-profile", "_explicit_options": {"profile"}},
    )

    assert yaml_config.profile == "yaml-profile"
    assert cli_config.profile == "cli-profile"


def test_invalid_spot_wait_is_user_facing(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    path.write_text(
        """
aws:
  dataset_s3_uri: s3://datasets/training.zip
  checkpoint_s3_uri: s3://checkpoints
  instance_type: ml.g4dn.xlarge
  max_runtime_seconds: 100
  max_wait_seconds: 99
training:
  model: yolo26
""",
        encoding="utf-8",
    )

    with pytest.raises(MLXUserError, match="max_wait_seconds"):
        load_aws_training_config(str(path), {"_explicit_options": set()})


def _write_checkpoint(path: Path, epoch: int, epochs: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "train_args": {"epochs": epochs},
            "optimizer": {"state": {}},
            "scaler": {},
            "ema": {},
        },
        path,
    )


def test_rotating_recovery_falls_back_when_newest_slot_is_corrupt(tmp_path: Path) -> None:
    work = tmp_path / "work"
    recovery = tmp_path / "recovery"
    last = work / "trial" / "weights" / "last.pt"
    publisher = RotatingCheckpointPublisher(
        work_dir=work,
        recovery_dir=recovery,
        provider="ultralytics",
        total_epochs=10,
    )

    _write_checkpoint(last, 0)
    first = publisher.publish_now()
    _write_checkpoint(last, 1)
    second = publisher.publish_now()
    assert first is not None and second is not None
    assert first.checkpoint_path.name == "resume-a.pt"
    assert second.checkpoint_path.name == "resume-b.pt"

    second.checkpoint_path.write_bytes(b"incomplete")
    restored = find_valid_recovery_checkpoint(
        recovery,
        expected_provider="ultralytics",
        required=True,
    )

    assert restored is not None
    assert restored.epoch == 1
    assert restored.checkpoint_path.name == "resume-a.pt"


def test_ultralytics_resume_copy_can_raise_total_epoch_target(tmp_path: Path) -> None:
    source = tmp_path / "resume-a.pt"
    _write_checkpoint(source, 3, epochs=10)
    recovery = type(
        "Recovery",
        (),
        {
            "checkpoint_path": source,
            "state_path": None,
            "epoch": 4,
            "provider": "ultralytics",
        },
    )()

    destination = prepare_working_resume_checkpoint(
        recovery,
        work_dir=tmp_path / "work",
        run_name="trial",
        provider="ultralytics",
        total_epochs=15,
    )

    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["train_args"]["epochs"] == 15
    assert torch.load(source, map_location="cpu", weights_only=False)["train_args"]["epochs"] == 10


def test_weights_only_checkpoint_is_not_accepted_for_recovery(tmp_path: Path) -> None:
    recovery = tmp_path / "recovery"
    checkpoint = recovery / "resume-a.pt"
    checkpoint.parent.mkdir()
    torch.save({"epoch": 2, "ema": {}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (recovery / "resume-a.json").write_text(
        json.dumps(
            {
                "provider": "ultralytics",
                "epoch": 3,
                "checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MLXUserError, match="No valid full-state"):
        find_valid_recovery_checkpoint(
            recovery,
            expected_provider="ultralytics",
            required=True,
        )


def test_libreyolo_full_state_checkpoint_is_recoverable(tmp_path: Path) -> None:
    work = tmp_path / "work"
    recovery = tmp_path / "recovery"
    last = work / "trial" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    torch.save(
        {
            "epoch": 4,
            "optimizer": {"state": {}},
            "model": {"weight": torch.tensor([1.0])},
            "config": {"epochs": 10},
        },
        last,
    )
    publisher = RotatingCheckpointPublisher(
        work_dir=work,
        recovery_dir=recovery,
        provider="libreyolo",
        total_epochs=10,
    )

    publisher.publish_now()
    restored = find_valid_recovery_checkpoint(
        recovery,
        expected_provider="libreyolo",
        required=True,
    )

    assert restored is not None
    assert restored.epoch == 5


def _status(name: str, state: str) -> AwsTrainingStatus:
    return AwsTrainingStatus(
        job_name=name,
        run_id="run",
        status=state,
        secondary_status=None,
        completed_epoch=None,
        total_epochs=None,
        progress_percent=None,
        elapsed_seconds=None,
        training_seconds=None,
        billable_seconds=None,
        managed_spot=True,
        spot_savings_percent=None,
        eta_seconds=None,
        expected_completion_time=None,
        interruptions=0,
        checkpoint_s3_uri=None,
        output_s3_uri=None,
        failure_reason=None,
        creation_time=None,
        start_time=None,
        end_time=None,
        console_url="https://example.test",
    )


def test_watch_reports_until_terminal_without_hidden_state() -> None:
    class FakeService:
        def __init__(self):
            self.values = iter([_status("job", "InProgress"), _status("job", "Completed")])

        def status(self, job_name):
            return next(self.values)

    observed = []
    result = WatchAwsObjectDetectionTraining(
        FakeService(),
        "job",
        poll_interval=1,
        on_status=observed.append,
        wait=lambda _: None,
    ).execute()

    assert result.status == "Completed"
    assert [item.status for item in observed] == ["InProgress", "Completed"]


def test_entrypoint_rejects_zip_traversal(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with zipfile.ZipFile(input_dir / "dataset.zip", "w") as archive:
        archive.writestr("../data.yaml", "names: {0: thing}\n")
    command = RunSageMakerObjectDetectionTraining(
        input_dir=input_dir,
        dataset_dir=tmp_path / "dataset",
    )

    with pytest.raises(MLXUserError, match="unsafe path"):
        command._extract_dataset()


def test_submit_payload_uses_spot_and_shared_run_prefix() -> None:
    class FakeS3:
        def __init__(self):
            self.objects = []

        def put_object(self, **kwargs):
            self.objects.append(kwargs)

    class FakeSageMaker:
        def __init__(self):
            self.request = None

        def create_training_job(self, **kwargs):
            self.request = kwargs
            return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/job"}

    config = AwsTrainingConfig(
        dataset_s3_uri="s3://datasets/data.zip",
        checkpoint_s3_uri="s3://shared/checkpoints",
        instance_type="ml.g4dn.xlarge",
        training=TrainObjectDetectionRequest(model="yolo26", device="auto"),
    )
    service = object.__new__(SageMakerTrainingService)
    service.config = config
    service.region = "us-east-1"
    service.s3 = FakeS3()
    service.sagemaker = FakeSageMaker()
    infrastructure = AwsInfrastructure(
        region="us-east-1",
        account_id="123",
        role_arn="arn:aws:iam::123:role/mlx",
        image_uri="123.dkr.ecr.us-east-1.amazonaws.com/mlx@sha256:abc",
    )

    result = service.submit(infrastructure, run_id="a" * 32)
    request = service.sagemaker.request

    assert request["EnableManagedSpotTraining"] is True
    assert request["StoppingCondition"]["MaxWaitTimeInSeconds"] == 172800
    assert request["CheckpointConfig"]["S3Uri"].endswith(
        "/mlx-od/runs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/recovery"
    )
    assert result.run_id == "a" * 32


def test_generated_execution_role_includes_vpc_network_permissions() -> None:
    class FakeIam:
        def __init__(self):
            self.policy = None

        def get_role(self, **kwargs):
            return {"Role": {"Arn": "arn:aws:iam::123:role/mlx"}}

        def put_role_policy(self, **kwargs):
            self.policy = json.loads(kwargs["PolicyDocument"])

    config = AwsTrainingConfig(
        dataset_s3_uri="s3://datasets/data.zip",
        checkpoint_s3_uri="s3://shared/checkpoints",
        instance_type="ml.g4dn.xlarge",
        training=TrainObjectDetectionRequest(model="yolo26"),
        vpc=AwsVpcConfig(
            subnet_ids=("subnet-123",),
            security_group_ids=("sg-123",),
        ),
    )
    service = object.__new__(SageMakerTrainingService)
    service.config = config
    service.iam = FakeIam()

    service._ensure_execution_role(
        repository_arn="arn:aws:ecr:us-east-1:123:repository/mlx"
    )

    statement = next(
        item
        for item in service.iam.policy["Statement"]
        if item["Sid"] == "TrainingVpcNetworkInterfaces"
    )
    assert "ec2:CreateNetworkInterface" in statement["Action"]
    assert "ec2:DeleteNetworkInterface" in statement["Action"]


def test_ultralytics_provider_reports_checkpoint_after_model_save(
    monkeypatch, tmp_path: Path
) -> None:
    from mlx.modes.object_detection.ultralytics import training

    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    events = []

    class FakeTraining:
        def __init__(self, config, *, checkpoint_observer=None):
            self.checkpoint_observer = checkpoint_observer

        def execute(self):
            self.checkpoint_observer(checkpoint)
            return "trained"

    monkeypatch.setattr(training, "TrainUltralyticsObjectDetection", FakeTraining)

    result = UltralyticsProvider().train(
        TrainObjectDetectionRequest(model="yolo26"),
        CallbackWorkflowReporter(events.append),
    )

    assert result == "trained"
    assert events[-1].payload == {"checkpoint_path": str(checkpoint)}
