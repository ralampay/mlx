from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.aws.checkpoints import (
    RotatingCheckpointPublisher,
    find_valid_recovery_checkpoint,
    prepare_working_resume_checkpoint,
)
from mlx.modes.image_classification.aws.config import (
    AwsTrainingConfig,
    load_aws_training_config,
    serialize_training_request,
)
from mlx.modes.image_classification.aws.entrypoint import (
    RunSageMakerImageClassificationTraining,
)
from mlx.modes.image_classification.aws.models import AwsInfrastructure
from mlx.modes.image_classification.aws.service import SageMakerTrainingService
from mlx.modes.image_classification.requests import ImageClassificationRequest


def _config(path: Path) -> None:
    path.write_text(
        """version: 1
aws:
  region: ap-southeast-1
  dataset_s3_uri: s3://datasets/classification.zip
  checkpoint_s3_uri: s3://training/checkpoints
  instance_type: ml.g4dn.xlarge
training:
  model: resnet18
  epochs: 5
  batch_size: 8
""",
        encoding="utf-8",
    )


def _checkpoint(path: Path, epoch: int, *, model: str = "resnet18") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "training_state_version": 1,
            "completed_epoch": epoch,
            "model_name": model,
            "family": "standard",
            "state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "history": [{"epoch": item} for item in range(1, epoch + 1)],
        },
        path,
    )


def test_config_defaults_to_spot_and_only_uses_explicit_cli_values(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _config(path)

    loaded = load_aws_training_config(
        str(path),
        {"epochs": 100, "batch_size": 1, "_explicit_options": {"epochs"}},
    )

    assert loaded.managed_spot is True
    assert loaded.resource_prefix == "mlx-ic"
    assert loaded.training.epochs == 100
    assert loaded.training.batch_size == 8
    assert loaded.training.device == "auto"
    assert loaded.training.input_size == (224, 224)


def test_config_rejects_non_training_fields(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _config(path)
    path.write_text(path.read_text(encoding="utf-8") + "  input_img: image.png\n", encoding="utf-8")

    with pytest.raises(MLXUserError, match="Unknown training"):
        load_aws_training_config(str(path), {"_explicit_options": set()})


def test_cli_dataset_s3_uri_overrides_classification_yaml(tmp_path: Path) -> None:
    path = tmp_path / "aws.yaml"
    _config(path)
    loaded = load_aws_training_config(
        str(path),
        {
            "dataset_s3_uri": "s3://portable/classification.zip",
            "_explicit_options": {"dataset_s3_uri"},
        },
    )
    assert loaded.dataset_s3_uri == "s3://portable/classification.zip"


def test_local_classification_does_not_load_aws_config(monkeypatch) -> None:
    from mlx.modes.image_classification import runner

    monkeypatch.setitem(runner.ACTION_HANDLERS, "test", lambda config: "local")

    assert runner.run_image_classification(
        {
            "platform": "local",
            "action": "test",
            "model": "resnet18",
            "width": 224,
            "height": 224,
        }
    ) == "local"


def test_rotating_recovery_falls_back_and_restores_best(tmp_path: Path) -> None:
    output, recovery = tmp_path / "output", tmp_path / "recovery"
    last, best = output / "resnet18.last.pth", output / "resnet18.pth"
    _checkpoint(last, 1)
    _checkpoint(best, 1)
    publisher = RotatingCheckpointPublisher(
        recovery_dir=recovery,
        output_dir=output,
        model_name="resnet18",
        compatibility_fingerprint="fingerprint",
        image_uri="image@sha256:abc",
        total_epochs=5,
    )
    first = publisher.publish_path(last)
    _checkpoint(last, 2)
    second = publisher.publish_path(last)
    assert first and second
    second.checkpoint_path.write_bytes(b"corrupt")

    restored = find_valid_recovery_checkpoint(
        recovery,
        required=True,
        expected_fingerprint="fingerprint",
        expected_image_uri="image@sha256:abc",
    )
    destination = prepare_working_resume_checkpoint(restored, output_dir=tmp_path / "resumed")

    assert restored.epoch == 1
    assert destination.name == "resnet18.last.pth"
    assert (tmp_path / "resumed" / "resnet18.pth").is_file()


def test_weights_only_checkpoint_is_not_recoverable(tmp_path: Path) -> None:
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    checkpoint = recovery / "resume-a.pth"
    torch.save({"state_dict": {}}, checkpoint)
    (recovery / "resume-a.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MLXUserError, match="No valid full-state"):
        find_valid_recovery_checkpoint(recovery, required=True)


def test_entrypoint_extracts_wrapped_dataset_and_rejects_traversal(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with zipfile.ZipFile(input_dir / "dataset.zip", "w") as archive:
        archive.writestr("animals/train/cat/a.jpg", b"image")
        archive.writestr("animals/val/cat/a.jpg", b"image")
    command = RunSageMakerImageClassificationTraining(
        input_dir=input_dir, dataset_dir=tmp_path / "dataset"
    )
    assert command._extract_dataset(max_uncompressed_bytes=1000).name == "animals"

    with zipfile.ZipFile(input_dir / "dataset.zip", "w") as archive:
        archive.writestr("../train/cat/a.jpg", b"image")
    with pytest.raises(MLXUserError, match="unsafe path"):
        command._extract_dataset(max_uncompressed_bytes=1000)


@pytest.mark.parametrize(
    ("training", "expected_model", "expected_ood"),
    [
        ({"model": "resnet18", "epochs": 1}, "resnet18", "none"),
        (
            {"model": "resnet18", "epochs": 1, "ood_method": "deep-svdd"},
            "resnet18",
            "deep-svdd",
        ),
        ({"model": "siamese-le-net", "epochs": 1}, "siamese-le-net", "none"),
    ],
)
def test_entrypoint_invokes_existing_trainer_for_every_family(
    monkeypatch,
    tmp_path: Path,
    training: dict,
    expected_model: str,
    expected_ood: str,
) -> None:
    from mlx.modes.image_classification.aws import entrypoint

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with zipfile.ZipFile(input_dir / "dataset.zip", "w") as archive:
        archive.writestr("train/cat/a.jpg", b"image")
        archive.writestr("val/cat/a.jpg", b"image")
    hyperparameters = tmp_path / "hyperparameters.json"
    hyperparameters.write_text(
        json.dumps(
            {
                "mlx_training": json.dumps(training),
                "mlx_volume_size_gb": "1",
                "mlx_resume": "false",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeTraining:
        def __init__(self, request, *, reporter):
            captured.update(request.to_config())

        def execute(self):
            output = Path(captured["output_path"])
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{captured['model']}.pth").touch()
            (output / f"{captured['model']}.last.pth").touch()
            (output / "training.csv").write_text("epoch\n", encoding="utf-8")
            return None

    monkeypatch.setattr(entrypoint, "TrainImageClassificationModel", FakeTraining)

    entrypoint.RunSageMakerImageClassificationTraining(
        hyperparameters_path=hyperparameters,
        input_dir=input_dir,
        checkpoint_dir=tmp_path / "checkpoints",
        model_dir=tmp_path / "model",
        work_dir=tmp_path / "work",
        dataset_dir=tmp_path / "dataset",
    ).execute()

    assert captured["model"] == expected_model
    assert captured["ood_method"] == expected_ood
    assert captured["dataset_path"].endswith("dataset")
    assert captured["output_path"].endswith("work/artifacts")


def test_submit_uses_spot_and_classification_prefix() -> None:
    class FakeS3:
        def __init__(self): self.objects = []
        def put_object(self, **kwargs): self.objects.append(kwargs)

    class FakeSageMaker:
        request = None
        def create_training_job(self, **kwargs):
            self.request = kwargs
            return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/job"}

    config = AwsTrainingConfig(
        dataset_s3_uri="s3://datasets/data.zip",
        checkpoint_s3_uri="s3://training/checkpoints",
        instance_type="ml.g4dn.xlarge",
        training=ImageClassificationRequest(model="resnet18", device="auto"),
    )
    service = object.__new__(SageMakerTrainingService)
    service.config, service.region, service.s3, service.sagemaker = config, "us-east-1", FakeS3(), FakeSageMaker()
    result = service.submit(
        AwsInfrastructure("us-east-1", "123", "arn:aws:iam::123:role/mlx", "image@sha256:abc"),
        run_id="a" * 32,
    )

    request = service.sagemaker.request
    assert request["EnableManagedSpotTraining"] is True
    assert request["CheckpointConfig"]["S3Uri"].endswith(f"/mlx-ic/runs/{'a' * 32}/recovery")
    assert result.run_id == "a" * 32


def test_resume_preserves_payload_and_only_raises_epoch_target() -> None:
    original = serialize_training_request(
        ImageClassificationRequest(model="resnet18", device="auto", epochs=5)
    )
    original.pop("svdd_warmup_epochs")
    config = AwsTrainingConfig(
        dataset_s3_uri="s3://datasets/data.zip",
        checkpoint_s3_uri="s3://training/checkpoints",
        instance_type="ml.g4dn.xlarge",
        training=ImageClassificationRequest(model="resnet18", device="auto", epochs=8),
    )
    service = object.__new__(SageMakerTrainingService)
    service.config, service.region = config, "us-east-1"
    service._describe = lambda _: {
        "TrainingJobStatus": "Stopped",
        "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/old",
        "HyperParameters": {"mlx_run_id": "run", "mlx_run_spec_s3_uri": "s3://spec"},
    }
    service._ensure_no_active_attempt = lambda *args, **kwargs: None
    service._get_json = lambda _: {
        "dataset_s3_uri": config.dataset_s3_uri,
        "checkpoint_base_s3_uri": config.checkpoint_s3_uri,
        "run_base_s3_uri": "s3://training/checkpoints/mlx-ic/runs/run",
        "image_uri": "image@sha256:abc",
        "role_arn": "arn:aws:iam::123:role/mlx",
        "resource_prefix": "mlx-ic",
        "training": original,
    }
    service._latest_recovery_epoch = lambda _: 4
    submitted = {}
    service.submit = lambda infrastructure, **kwargs: submitted.update(kwargs) or "submitted"

    assert service.resume("old") == "submitted"
    assert submitted["training_payload"] == {**original, "epochs": 8}
    assert "svdd_warmup_epochs" not in submitted["training_payload"]
