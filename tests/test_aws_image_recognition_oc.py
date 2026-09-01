from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from mlx.core.aws.models import AwsInfrastructure
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.artifacts import resolve_training_paths
from mlx.modes.image_recognition_oc.aws.config import (
    AwsImageOneClassConfig,
    load_aws_config,
)
from mlx.modes.image_recognition_oc.aws.entrypoint import RunSageMakerImageOneClass
from mlx.modes.image_recognition_oc.aws.service import SageMakerImageOneClassService
from mlx.modes.image_recognition_oc.variants import image_one_class_svdd_variants


def _config(path: Path, *, backbone: str = "resnet18", benchmark: bool = True) -> None:
    backbone_line = f"  backbone: {backbone}\n" if backbone else ""
    path.write_text(
        f"""version: 1
aws:
  region: ap-southeast-1
  dataset_s3_uri: s3://datasets/occ-bottle.zip
  output_s3_uri: s3://results/one-class
  resource_prefix: mvtec-bottle
  instance_type: ml.g5.xlarge
training:
  model: deep-svdd
{backbone_line}  epochs: 5
  batch_size: 4
benchmark:
  enabled: {str(benchmark).lower()}
  batch_size: 8
""",
        encoding="utf-8",
    )


def test_config_supports_single_all_and_standalone_benchmark(tmp_path: Path) -> None:
    single = tmp_path / "single.yaml"
    _config(single)
    loaded = load_aws_config(str(single), {"action": "train", "_explicit_options": set()})
    assert loaded.output_s3_uri == "s3://results/one-class"
    assert loaded.resource_prefix == "mvtec-bottle"
    assert loaded.training["backbone"] == "resnet18"
    assert loaded.training["device"] == "auto"
    assert loaded.benchmark == {
        "enabled": True,
        "batch_size": 8,
        "workers": 0,
        "plots": True,
    }

    batch = tmp_path / "batch.yaml"
    _config(batch, backbone="")
    loaded_batch = load_aws_config(
        str(batch), {"action": "train-all", "_explicit_options": set()}
    )
    assert "backbone" not in loaded_batch.training

    with pytest.raises(MLXUserError, match="omit training.backbone"):
        load_aws_config(
            str(single), {"action": "train-all", "_explicit_options": set()}
        )

    missing_backbone = tmp_path / "missing-backbone.yaml"
    _config(missing_backbone, backbone="")
    with pytest.raises(MLXUserError, match="requires training.backbone"):
        load_aws_config(
            str(missing_backbone), {"action": "train", "_explicit_options": set()}
        )

    benchmark_config = tmp_path / "benchmark.yaml"
    _config(benchmark_config)
    text = benchmark_config.read_text(encoding="utf-8").replace(
        "  batch_size: 8\n", "  batch_size: 8\n  model_s3_uri: s3://models/resnet18-deep-svdd.pth\n"
    )
    benchmark_config.write_text(text, encoding="utf-8")
    loaded_benchmark = load_aws_config(
        str(benchmark_config), {"action": "benchmark", "_explicit_options": set()}
    )
    assert loaded_benchmark.benchmark["model_s3_uri"].endswith(".pth")
    loaded_status = load_aws_config(
        str(benchmark_config), {"action": "status", "_explicit_options": set()}
    )
    assert loaded_status.benchmark["model_s3_uri"].endswith(".pth")


def test_svdd_inventory_has_every_backbone_and_both_drax_fusions() -> None:
    variants = image_one_class_svdd_variants()

    assert len(variants) == 13
    assert len({item.variant_id for item in variants}) == 13
    assert all(item.model_name == "deep-svdd" for item in variants)
    assert {item.variant_id for item in variants if item.backbone_name == "draxnet"} == {
        "draxnet-average-deep-svdd",
        "draxnet-sknet-deep-svdd",
    }
    assert not any("siamese" in item.backbone_name for item in variants)


def test_documented_example_config_is_valid() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/image_recognition_oc/aws-training.example.yaml"
    )

    loaded = load_aws_config(str(path), {"action": "train", "_explicit_options": set()})

    assert loaded.instance_type == "ml.g5.xlarge"
    assert loaded.output_s3_uri == "s3://my-results/one-class"
    assert loaded.resource_prefix == "mvtec-bottle-svdd"


def _service(config: AwsImageOneClassConfig):
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

    service = object.__new__(SageMakerImageOneClassService)
    service.config = config
    service.region = "us-east-1"
    service.s3 = FakeS3()
    service.sagemaker = FakeSageMaker()
    service._client_error = RuntimeError
    service._boto_error = OSError
    service.variant_factory = image_one_class_svdd_variants
    return service


def test_train_all_submission_freezes_inventory_and_result_prefix() -> None:
    config = AwsImageOneClassConfig(
        action="train-all",
        dataset_s3_uri="s3://datasets/occ.zip",
        output_s3_uri="s3://results/project",
        instance_type="ml.g5.xlarge",
        training={"model": "deep-svdd", "epochs": 5, "batch_size": 4},
        benchmark={"enabled": True, "batch_size": 8, "workers": 0, "plots": True},
        resource_prefix="occ-bottle",
    )
    service = _service(config)

    result = service.submit(
        AwsInfrastructure("us-east-1", "123", "arn:aws:iam::123:role/mlx", "image@sha256:abc"),
        run_id="a" * 32,
    )

    request = service.sagemaker.request
    assert result.variant_count == 13
    assert result.run_s3_uri == f"s3://results/project/occ-bottle/runs/{'a' * 32}"
    assert request["CheckpointConfig"]["S3Uri"] == result.run_s3_uri
    assert len(json.loads(request["HyperParameters"]["mlx_variants"])) == 13
    assert [channel["ChannelName"] for channel in request["InputDataConfig"]] == ["training"]
    assert len(service.s3.objects) == 2


def test_standalone_benchmark_submission_has_model_channel() -> None:
    config = AwsImageOneClassConfig(
        action="benchmark",
        dataset_s3_uri="s3://datasets/occ.zip",
        output_s3_uri="s3://results/project",
        instance_type="ml.g5.xlarge",
        training={"model": "deep-svdd"},
        benchmark={
            "enabled": False,
            "batch_size": 8,
            "workers": 0,
            "plots": True,
            "model_s3_uri": "s3://models/resnet18-deep-svdd.pth",
        },
        resource_prefix="occ-bottle",
    )
    service = _service(config)

    result = service.submit(
        AwsInfrastructure("us-east-1", "123", "arn:aws:iam::123:role/mlx", "image@sha256:abc"),
        run_id="b" * 32,
    )

    assert result.variant_count == 0
    assert "/benchmarks/" in result.run_s3_uri
    assert [channel["ChannelName"] for channel in service.sagemaker.request["InputDataConfig"]] == [
        "training",
        "model",
    ]


def test_status_and_stop_report_variant_progress() -> None:
    manifest = {
        "current_variant": "resnet50-deep-svdd",
        "variants": [
            {
                "variant_id": "resnet18-deep-svdd",
                "backbone_name": "resnet18",
                "drax_fusion_mode": None,
                "status": "completed",
                "benchmark_status": "completed",
                "completed_epoch": 5,
                "total_epochs": 5,
            },
            {
                "variant_id": "resnet50-deep-svdd",
                "backbone_name": "resnet50",
                "drax_fusion_mode": None,
                "status": "running",
                "benchmark_status": "pending",
                "completed_epoch": 2,
                "total_epochs": 5,
            },
        ],
    }

    class Body:
        def read(self):
            return json.dumps(manifest).encode()

    class FakeS3:
        def get_object(self, **_kwargs):
            return {"Body": Body()}

    class FakeSageMaker:
        stopped = None

        def describe_training_job(self, **_kwargs):
            return {
                "TrainingJobName": "job",
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/job",
                "TrainingJobStatus": "InProgress",
                "HyperParameters": {
                    "mlx_run_id": "run",
                    "mlx_operation": "train-all",
                },
                "CheckpointConfig": {"S3Uri": "s3://results/prefix/runs/run"},
                "OutputDataConfig": {"S3OutputPath": "s3://results/output"},
            }

        def stop_training_job(self, **kwargs):
            self.stopped = kwargs["TrainingJobName"]

    service = object.__new__(SageMakerImageOneClassService)
    service.config = AwsImageOneClassConfig(
        action="status",
        dataset_s3_uri="s3://datasets/occ.zip",
        output_s3_uri="s3://results",
        instance_type="ml.g5.xlarge",
        training={"model": "deep-svdd"},
        benchmark={"enabled": True},
    )
    service.region = "us-east-1"
    service.s3 = FakeS3()
    service.sagemaker = FakeSageMaker()
    service._client_error = RuntimeError
    service._boto_error = OSError

    status = service.status("job")
    stopped = service.stop("job", config_path="aws.yaml")

    assert status.completed_variants == 1
    assert status.total_variants == 2
    assert status.progress_percent == pytest.approx(70.0)
    assert stopped.status == "Stopping"
    assert service.sagemaker.stopped == "job"
    assert "--action resume" in stopped.next_command


def _dataset_zip(input_dir: Path) -> None:
    input_dir.mkdir()
    with zipfile.ZipFile(input_dir / "dataset.zip", "w") as archive:
        archive.writestr("occ/train/normal/one.png", b"image")
        archive.writestr("occ/val/normal/one.png", b"image")
        archive.writestr("occ/test/normal/one.png", b"image")
        archive.writestr("occ/test/anomaly/crack/one.png", b"image")


def _hyperparameters(path: Path, variants: list[dict], *, operation="train-all", benchmark=True) -> None:
    path.write_text(
        json.dumps(
            {
                "mlx_run_id": "run-1",
                "mlx_operation": operation,
                "mlx_training": json.dumps(
                    {"model": "deep-svdd", "epochs": 2, "batch_size": 1, "lr": 0.001}
                ),
                "mlx_benchmark": json.dumps(
                    {"enabled": benchmark, "batch_size": 1, "workers": 0, "plots": False}
                ),
                "mlx_variants": json.dumps(variants),
                "mlx_volume_size_gb": "1",
                "mlx_image_uri": "image@sha256:abc",
            }
        ),
        encoding="utf-8",
    )


class FakeTrainer:
    calls = []

    def __init__(self, request, *, reporter):
        self.request = request

    def execute(self):
        config = self.request.to_config()
        self.calls.append((config["backbone"], config["drax_fusion_mode"]))
        paths = resolve_training_paths(config)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        base = {
            "mode": "image_recognition_oc",
            "checkpoint_version": 1,
            "model_name": "deep-svdd",
            "backbone_name": config["backbone"],
            "state_dict": {},
        }
        torch.save(base, paths["checkpoint"])
        torch.save(
            {
                **base,
                "training_state_version": 1,
                "completed_epoch": 2,
                "optimizer_state_dict": {},
                "history": [{"epoch": 1}, {"epoch": 2}],
            },
            paths["last_checkpoint"],
        )
        paths["training_csv"].write_text("epoch\n1\n2\n", encoding="utf-8")
        paths["training_plot"].write_bytes(b"plot")
        paths["run_metadata"].write_text("{}", encoding="utf-8")
        return {"paths": paths}


class FakeBenchmark:
    calls = []

    def __init__(self, request):
        self.request = request

    def execute(self):
        config = self.request.to_config()
        self.calls.append(config["model_path"])
        output = Path(config["output_path"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text('{"auroc": 1.0}', encoding="utf-8")
        (output / "predictions.csv").write_text("image\n", encoding="utf-8")
        (output / "run_metadata.json").write_text("{}", encoding="utf-8")
        return {"metrics": {"auroc": 1.0}}


def test_entrypoint_trains_variants_and_benchmarks_each(tmp_path: Path) -> None:
    FakeTrainer.calls = []
    FakeBenchmark.calls = []
    input_dir = tmp_path / "input"
    _dataset_zip(input_dir)
    hp = tmp_path / "hyperparameters.json"
    variants = [
        {"model_name": "deep-svdd", "backbone_name": "resnet18", "variant_id": "resnet18-deep-svdd", "drax_fusion_mode": None},
        {"model_name": "deep-svdd", "backbone_name": "draxnet", "variant_id": "draxnet-sknet-deep-svdd", "drax_fusion_mode": "sknet"},
    ]
    _hyperparameters(hp, variants)
    checkpoints = tmp_path / "checkpoints"

    result = RunSageMakerImageOneClass(
        hyperparameters_path=hp,
        input_dir=input_dir,
        checkpoint_dir=checkpoints,
        model_dir=tmp_path / "model",
        work_dir=tmp_path / "work",
        dataset_dir=tmp_path / "dataset",
        trainer_factory=FakeTrainer,
        benchmark_factory=FakeBenchmark,
    ).execute()

    assert FakeTrainer.calls == [("resnet18", "average"), ("draxnet", "sknet")]
    assert len(FakeBenchmark.calls) == 2
    assert result["status"] == "completed"
    assert all(item["benchmark_status"] == "completed" for item in result["variants"])
    assert (checkpoints / "models/resnet18-deep-svdd/benchmark/metrics.json").is_file()

    RunSageMakerImageOneClass(
        hyperparameters_path=hp,
        input_dir=input_dir,
        checkpoint_dir=checkpoints,
        model_dir=tmp_path / "model-resume",
        work_dir=tmp_path / "work-resume",
        dataset_dir=tmp_path / "dataset-resume",
        trainer_factory=FakeTrainer,
        benchmark_factory=FakeBenchmark,
    ).execute()
    assert len(FakeTrainer.calls) == 2
    assert len(FakeBenchmark.calls) == 2


def test_entrypoint_runs_standalone_benchmark(tmp_path: Path) -> None:
    FakeBenchmark.calls = []
    input_dir = tmp_path / "input"
    _dataset_zip(input_dir)
    model_input = tmp_path / "model-input"
    model_input.mkdir()
    torch.save(
        {
            "mode": "image_recognition_oc",
            "checkpoint_version": 1,
            "model_name": "deep-svdd",
            "backbone_name": "resnet18",
            "state_dict": {},
        },
        model_input / "model.pth",
    )
    hp = tmp_path / "hyperparameters.json"
    _hyperparameters(hp, [], operation="benchmark", benchmark=False)

    result = RunSageMakerImageOneClass(
        hyperparameters_path=hp,
        input_dir=input_dir,
        model_input_dir=model_input,
        checkpoint_dir=tmp_path / "checkpoints",
        model_dir=tmp_path / "model",
        work_dir=tmp_path / "work",
        dataset_dir=tmp_path / "dataset",
        benchmark_factory=FakeBenchmark,
    ).execute()

    assert result["manifest"]["status"] == "completed"
    assert result["metrics"] == {"auroc": 1.0}
    assert (tmp_path / "model/artifacts/metrics.json").is_file()


def test_resume_retries_failed_benchmark_without_retraining(tmp_path: Path) -> None:
    FakeTrainer.calls = []
    input_dir = tmp_path / "input"
    _dataset_zip(input_dir)
    hp = tmp_path / "hyperparameters.json"
    variants = [
        {
            "model_name": "deep-svdd",
            "backbone_name": "resnet18",
            "variant_id": "resnet18-deep-svdd",
            "drax_fusion_mode": None,
        }
    ]
    _hyperparameters(hp, variants)

    class FailOnceBenchmark(FakeBenchmark):
        attempts = 0

        def execute(self):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise MLXUserError("benchmark interrupted")
            return super().execute()

    checkpoints = tmp_path / "checkpoints"
    first = RunSageMakerImageOneClass(
        hyperparameters_path=hp,
        input_dir=input_dir,
        checkpoint_dir=checkpoints,
        model_dir=tmp_path / "model-first",
        work_dir=tmp_path / "work-first",
        dataset_dir=tmp_path / "dataset-first",
        trainer_factory=FakeTrainer,
        benchmark_factory=FailOnceBenchmark,
    )
    with pytest.raises(MLXUserError, match="benchmark interrupted"):
        first.execute()

    failed = json.loads((checkpoints / "run-status.json").read_text(encoding="utf-8"))
    assert failed["variants"][0]["status"] == "completed"
    assert failed["variants"][0]["benchmark_status"] == "failed"

    RunSageMakerImageOneClass(
        hyperparameters_path=hp,
        input_dir=input_dir,
        checkpoint_dir=checkpoints,
        model_dir=tmp_path / "model-resumed",
        work_dir=tmp_path / "work-resumed",
        dataset_dir=tmp_path / "dataset-resumed",
        trainer_factory=FakeTrainer,
        benchmark_factory=FailOnceBenchmark,
    ).execute()

    assert len(FakeTrainer.calls) == 1
    assert FailOnceBenchmark.attempts == 2


def test_runner_routes_aws_without_changing_local_actions(monkeypatch) -> None:
    from mlx.modes.image_recognition_oc import runner
    from mlx.modes.image_recognition_oc.aws import runner as aws_runner

    monkeypatch.setattr(
        aws_runner,
        "run_aws_image_one_class",
        lambda config: (config["platform"], config["action"]),
    )

    assert runner.run_image_recognition_oc(
        {"platform": "aws", "action": "train-all"}
    ) == ("aws", "train-all")
