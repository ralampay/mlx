from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXAbort, MLXUserError
from mlx.modes.object_detection import runner as detection_runner
from mlx.modes.object_detection.aws import runner as aws_detection_runner
from mlx.modes.object_detection.aws.checkpoints import (
    RotatingCheckpointPublisher,
    find_valid_recovery_checkpoint,
    prepare_working_resume_checkpoint,
)
from mlx.modes.object_detection.aws.commands import (
    LocateBestAwsObjectDetectionModel,
    WatchAwsObjectDetectionTraining,
)
from mlx.modes.object_detection.aws.config import (
    AwsTrainingConfig,
    AwsVpcConfig,
    load_aws_training_config,
)
from mlx.modes.object_detection.aws.entrypoint import RunSageMakerObjectDetectionTraining
from mlx.modes.object_detection.aws.image import PublishSageMakerImage
from mlx.modes.object_detection.aws.models import (
    AwsBestModelLocation,
    AwsInfrastructure,
    AwsTrainingStatus,
)
from mlx.modes.object_detection.aws.service import SageMakerTrainingService
from mlx.modes.object_detection.aws.status import build_training_status
from mlx.modes.object_detection.aws.presentation import render_aws_result
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


def test_explicit_cli_dataset_s3_uri_overrides_yaml_for_new_run(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _write_config(path)

    config = load_aws_training_config(
        str(path),
        {
            "dataset_s3_uri": "s3://portable/override.zip",
            "_explicit_options": {"dataset_s3_uri"},
        },
    )

    assert config.dataset_s3_uri == "s3://portable/override.zip"


def test_aws_dataset_cli_override_rejects_explicit_local_dataset(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _write_config(path)
    with pytest.raises(MLXUserError, match="either --dataset"):
        load_aws_training_config(
            str(path),
            {
                "dataset_path": "./local",
                "dataset_s3_uri": "s3://portable/override.zip",
                "_explicit_options": {"dataset_path", "dataset_s3_uri"},
            },
        )

    with pytest.raises(MLXUserError, match="AWS train or resume"):
        load_aws_training_config(
            str(path),
            {
                "action": "status",
                "dataset_s3_uri": "s3://portable/override.zip",
                "_explicit_options": {"dataset_s3_uri"},
            },
        )


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


def test_watch_translates_keyboard_interrupt_to_intentional_abort() -> None:
    class FakeService:
        def status(self, job_name):
            return _status(job_name, "InProgress")

    observed = []

    def interrupt_wait(_: float) -> None:
        raise KeyboardInterrupt

    command = WatchAwsObjectDetectionTraining(
        FakeService(),
        "job",
        poll_interval=1,
        on_status=observed.append,
        wait=interrupt_wait,
    )

    with pytest.raises(MLXAbort):
        command.execute()

    assert [item.status for item in observed] == ["InProgress"]


class _FakeAwsClientError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _FakeAwsBotoError(Exception):
    pass


class _FakeBestModelS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, *, Bucket, Key):
        value = self.objects.get((Bucket, Key))
        if value is None:
            raise _FakeAwsClientError("NoSuchKey")
        return {"Body": io.BytesIO(json.dumps(value).encode("utf-8"))}

    def head_object(self, *, Bucket, Key):
        value = self.objects.get((Bucket, Key))
        if value is None:
            raise _FakeAwsClientError("NoSuchKey")
        return value


class _FakeTrainingJobsPaginator:
    def __init__(self, summaries):
        self.summaries = summaries
        self.arguments = None

    def paginate(self, **kwargs):
        self.arguments = kwargs
        return [
            {"TrainingJobSummaries": [summary]}
            for summary in self.summaries
        ]


class _FakeBestModelSageMaker:
    def __init__(self, descriptions):
        self.descriptions = descriptions
        self.paginator = _FakeTrainingJobsPaginator([
            {"TrainingJobName": name} for name in descriptions
        ])

    def get_paginator(self, name):
        assert name == "list_training_jobs"
        return self.paginator

    def describe_training_job(self, *, TrainingJobName):
        return self.descriptions[TrainingJobName]


def _best_model_service(*, descriptions, objects, profile=None):
    config = AwsTrainingConfig(
        dataset_s3_uri="s3://datasets/data.zip",
        checkpoint_s3_uri="s3://checkpoints",
        instance_type="ml.g4dn.xlarge",
        training=TrainObjectDetectionRequest(
            provider="libreyolo",
            model="yolo9-s-drax-b5",
        ),
        region="ap-southeast-1",
        profile=profile,
        resource_prefix="mlx-od-visdrone",
    )
    service = object.__new__(SageMakerTrainingService)
    service.config = config
    service.region = "ap-southeast-1"
    service.s3 = _FakeBestModelS3(objects)
    service.sagemaker = _FakeBestModelSageMaker(descriptions)
    service._client_error = _FakeAwsClientError
    service._boto_error = _FakeAwsBotoError
    return service


def _run_artifact_objects(
    run_id,
    *,
    model="yolo9-s-drax-b5",
    dataset_s3_uri="s3://datasets/data.zip",
    include_best=True,
):
    base_key = f"mlx-od-visdrone/runs/{run_id}"
    spec_uri = f"s3://checkpoints/{base_key}/run-spec.json"
    values = {
        ("checkpoints", f"{base_key}/run-spec.json"): {
            "run_id": run_id,
            "run_base_s3_uri": f"s3://checkpoints/{base_key}",
            "dataset_s3_uri": dataset_s3_uri,
            "checkpoint_base_s3_uri": "s3://checkpoints",
            "resource_prefix": "mlx-od-visdrone",
            "training": {"provider": "libreyolo", "model": model},
        },
    }
    if include_best:
        values.update({
            ("checkpoints", f"{base_key}/recovery/best.json"): {
                "provider": "libreyolo",
                "checkpoint_sha256": "a" * 64,
            },
            ("checkpoints", f"{base_key}/recovery/best.pt"): {
                "ContentLength": 123456,
                "LastModified": datetime(2026, 8, 27, tzinfo=timezone.utc),
            },
            ("checkpoints", f"{base_key}/recovery/current.json"): {
                "epoch": 100,
            },
        })
    return spec_uri, values


def test_best_model_locator_returns_newest_completed_downloadable_match() -> None:
    missing_id = "1" * 32
    selected_id = "2" * 32
    unrelated_id = "3" * 32
    unrelated_dataset_id = "7" * 32
    missing_spec, missing_objects = _run_artifact_objects(missing_id, include_best=False)
    selected_spec, selected_objects = _run_artifact_objects(selected_id)
    unrelated_spec, unrelated_objects = _run_artifact_objects(
        unrelated_id,
        model="yolo9-t",
    )
    unrelated_dataset_spec, unrelated_dataset_objects = _run_artifact_objects(
        unrelated_dataset_id,
        dataset_s3_uri="s3://datasets/other.zip",
    )
    descriptions = {
        "newest-without-best": {
            "TrainingJobStatus": "Completed",
            "HyperParameters": {
                "mlx_run_id": missing_id,
                "mlx_run_spec_s3_uri": missing_spec,
            },
        },
        "unrelated-model": {
            "TrainingJobStatus": "Completed",
            "HyperParameters": {
                "mlx_run_id": unrelated_id,
                "mlx_run_spec_s3_uri": unrelated_spec,
            },
        },
        "unrelated-dataset": {
            "TrainingJobStatus": "Completed",
            "HyperParameters": {
                "mlx_run_id": unrelated_dataset_id,
                "mlx_run_spec_s3_uri": unrelated_dataset_spec,
            },
        },
        "selected": {
            "TrainingJobStatus": "Completed",
            "HyperParameters": {
                "mlx_run_id": selected_id,
                "mlx_run_spec_s3_uri": selected_spec,
            },
        },
    }
    service = _best_model_service(
        descriptions=descriptions,
        objects={
            **missing_objects,
            **selected_objects,
            **unrelated_objects,
            **unrelated_dataset_objects,
        },
        profile="mlx-training",
    )

    result = LocateBestAwsObjectDetectionModel(service).execute()

    assert result.job_name == "selected"
    assert result.run_id == selected_id
    assert result.completed_epoch == 100
    assert result.size_bytes == 123456
    assert result.sha256 == "a" * 64
    assert result.best_model_s3_uri.endswith(f"/{selected_id}/recovery/best.pt")
    assert result.download_command.endswith(
        "--region ap-southeast-1 --profile mlx-training"
    )
    assert service.sagemaker.paginator.arguments == {
        "StatusEquals": "Completed",
        "NameContains": "mlx-od-visdrone",
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
    }


def test_best_model_locator_reports_matching_runs_without_valid_best() -> None:
    run_id = "4" * 32
    spec_uri, objects = _run_artifact_objects(run_id, include_best=False)
    service = _best_model_service(
        descriptions={
            "completed": {
                "TrainingJobStatus": "Completed",
                "HyperParameters": {
                    "mlx_run_id": run_id,
                    "mlx_run_spec_s3_uri": spec_uri,
                },
            }
        },
        objects=objects,
    )

    with pytest.raises(MLXUserError, match="none has a valid downloadable"):
        service.locate_best_model()


def test_best_model_locator_reports_no_completed_matching_run() -> None:
    service = _best_model_service(
        descriptions={
            "unmanaged": {
                "TrainingJobStatus": "Completed",
                "HyperParameters": {},
            }
        },
        objects={},
    )

    with pytest.raises(MLXUserError, match="No completed AWS object-detection run"):
        service.locate_best_model()


def test_best_model_locator_translates_aws_listing_failures() -> None:
    service = _best_model_service(descriptions={}, objects={})

    class FailingSageMaker:
        def get_paginator(self, name):
            raise _FakeAwsBotoError("unavailable")

    service.sagemaker = FailingSageMaker()

    with pytest.raises(MLXUserError, match="AWS best-model lookup failed"):
        service.locate_best_model()


def test_best_model_action_does_not_require_job_name(monkeypatch) -> None:
    location = AwsBestModelLocation(
        job_name="job",
        run_id="5" * 32,
        provider="libreyolo",
        model="yolo9-s-drax-b5",
        completed_epoch=100,
        best_model_s3_uri="s3://checkpoints/run/recovery/best.pt",
        size_bytes=123,
        last_modified=None,
        sha256="b" * 64,
        download_command="aws s3 cp source target",
    )

    class FakeService:
        def __init__(self, config):
            self.config = config

        def locate_best_model(self):
            return location

    rendered = []
    monkeypatch.setattr(aws_detection_runner, "load_aws_training_config", lambda *_: "config")
    monkeypatch.setattr(aws_detection_runner, "SageMakerTrainingService", FakeService)
    monkeypatch.setattr(
        aws_detection_runner,
        "render_aws_result",
        lambda value, **_: rendered.append(value),
    )

    result = aws_detection_runner.run_aws_object_detection(
        {"action": "best-model", "config_path": "aws.yaml"}
    )

    assert result is location
    assert rendered == [location]


def test_best_model_location_renders_machine_readable_json(capsys) -> None:
    location = AwsBestModelLocation(
        job_name="job",
        run_id="6" * 32,
        provider="libreyolo",
        model="yolo9-s-drax-b5",
        completed_epoch=100,
        best_model_s3_uri="s3://checkpoints/run/recovery/best.pt",
        size_bytes=123,
        last_modified="2026-08-27T00:00:00+00:00",
        sha256="c" * 64,
        download_command="aws s3 cp source target",
    )

    render_aws_result(location, output_format="json")

    assert json.loads(capsys.readouterr().out) == location.to_dict()


def test_status_translation_is_independent_of_aws_clients() -> None:
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
    description = {
        "TrainingJobName": "job",
        "TrainingJobStatus": "InProgress",
        "CreationTime": now - timedelta(minutes=10),
        "TrainingStartTime": now - timedelta(minutes=9),
        "SecondaryStatusTransitions": [{"Status": "Training"}],
        "EnableManagedSpotTraining": True,
        "HyperParameters": {
            "mlx_run_id": "run",
            "mlx_training": json.dumps({"epochs": 20}),
        },
        "CheckpointConfig": {"S3Uri": "s3://checkpoints/run"},
        "OutputDataConfig": {"S3OutputPath": "s3://outputs/run"},
    }
    metrics = {"mlx:epoch": 5.0, "mlx:eta_seconds": 90.0}

    status = build_training_status(
        description,
        latest_metric=lambda _, name: metrics.get(name),
        console_url="https://example.test/job",
        now=now,
    )

    assert status.completed_epoch == 5
    assert status.total_epochs == 20
    assert status.progress_percent == 25.0
    assert status.elapsed_seconds == 600
    assert status.expected_completion_time == now + timedelta(seconds=90)


def test_image_publisher_reuses_digest_without_running_docker(tmp_path: Path) -> None:
    class FakeAwsError(Exception):
        pass

    class FakeEcr:
        def describe_images(self, **kwargs):
            return {"imageDetails": [{"imageDigest": "sha256:existing"}]}

    class FailingDocker:
        def login(self, **kwargs):
            raise AssertionError("Docker should not run for an existing source image")

        build = login
        push = login

    package_root = tmp_path / "package"
    package_root.mkdir()
    dockerfile = package_root / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    image_uri = PublishSageMakerImage(
        ecr=FakeEcr(),
        repository_name="mlx-training",
        repository_uri="123.dkr.ecr.us-east-1.amazonaws.com/mlx-training",
        package_root=package_root,
        dockerfile=dockerfile,
        rebuild=False,
        client_error=FakeAwsError,
        boto_error=FakeAwsError,
        command_runner=FailingDocker(),
    ).execute()

    assert image_uri.endswith("@sha256:existing")


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

    legacy_payload = config.training.to_config()
    for key in (
        "validate_after_training",
        "validation_split",
        "validation_confidence",
        "validation_iou",
        "validation_max_detections",
    ):
        legacy_payload.pop(key)
    service.submit(
        infrastructure,
        run_id="b" * 32,
        training_payload=legacy_payload,
    )

    assert json.loads(service.sagemaker.request["HyperParameters"]["mlx_training"]) == (
        legacy_payload
    )


def test_resume_preserves_original_training_payload_for_immutable_image() -> None:
    original_payload = TrainObjectDetectionRequest(
        provider="libreyolo",
        model="yolo9-s-drax-b5",
        epochs=100,
    ).to_config()
    for key in (
        "validate_after_training",
        "validation_split",
        "validation_confidence",
        "validation_iou",
        "validation_max_detections",
    ):
        original_payload.pop(key)

    config = AwsTrainingConfig(
        dataset_s3_uri="s3://datasets/data.zip",
        checkpoint_s3_uri="s3://shared/checkpoints",
        instance_type="ml.g4dn.xlarge",
        training=TrainObjectDetectionRequest(
            provider="libreyolo",
            model="yolo9-s-drax-b5",
            epochs=120,
        ),
    )
    service = object.__new__(SageMakerTrainingService)
    service.config = config
    service.region = "us-east-1"
    service._describe = lambda _: {
        "TrainingJobStatus": "Stopped",
        "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/old-job",
        "HyperParameters": {
            "mlx_run_id": "run-id",
            "mlx_run_spec_s3_uri": "s3://shared/checkpoints/run-spec.json",
        },
    }
    service._ensure_no_active_attempt = lambda *args, **kwargs: None
    service._get_json = lambda _: {
        "dataset_s3_uri": config.dataset_s3_uri,
        "checkpoint_base_s3_uri": config.checkpoint_s3_uri,
        "run_base_s3_uri": "s3://shared/checkpoints/mlx-od/runs/run-id",
        "image_uri": "123.dkr.ecr.us-east-1.amazonaws.com/mlx@sha256:old",
        "role_arn": "arn:aws:iam::123:role/mlx",
        "training": original_payload,
    }
    service._latest_recovery_epoch = lambda *args: 62
    submitted = {}

    def capture_submit(infrastructure, **kwargs):
        submitted.update(kwargs)
        return "submitted"

    service.submit = capture_submit

    result = service.resume("old-job")

    assert result == "submitted"
    assert submitted["training_payload"] == {**original_payload, "epochs": 120}
    assert "validate_after_training" not in submitted["training_payload"]
    assert submitted["training"].epochs == 120
    assert RunSageMakerObjectDetectionTraining._compatibility_fingerprint(
        submitted["training_payload"]
    ) == RunSageMakerObjectDetectionTraining._compatibility_fingerprint(original_payload)


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
