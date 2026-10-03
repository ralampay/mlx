from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.aws.comparison import (
    AdvanceDetectionComparison,
    GetDetectionComparisonStatus,
    LaunchDetectionComparisonTest,
    SubmitDetectionComparison,
    WatchDetectionComparison,
    comparison_models,
)
from mlx.modes.object_detection.aws.config import AwsTrainingConfig, load_aws_training_config
from mlx.modes.object_detection.aws.comparison_entrypoint import RunSageMakerDetectionComparisonTest
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)


class FakeService:
    def __init__(self, *, pretrained=False, managed_spot=False, provider="libreyolo"):
        self.config = AwsTrainingConfig(
            dataset_s3_uri="s3://data/quick.zip",
            checkpoint_s3_uri="s3://checkpoints",
            instance_type="ml.g4dn.2xlarge",
            resource_prefix="quick",
            managed_spot=managed_spot,
            training=TrainObjectDetectionRequest(
                provider=provider, model="yolox-m", pretrained=pretrained,
                validate_after_training=True, validation_split="val",
            ),
        )
        self.s3 = FakeS3()
        self.region = "ap-southeast-1"
        self.sagemaker = SimpleNamespace(create_training_job=self._create_job)
        self.submitted = []
        self.test_request = None
        self.job_status = "InProgress"
        self.fail_next_submit_quota = False

    def prepare_infrastructure(self):
        return SimpleNamespace(
            image_uri="image:tag",
            role_arn="arn:aws:iam::123456789012:role/test",
        )

    def submit(
        self, infrastructure, *, training, comparison_validation_s3_uri,
        run_id=None, job_name=None,
    ):
        if self.fail_next_submit_quota and self.submitted:
            self.fail_next_submit_quota = False
            raise MLXUserError("AWS training job submission failed: ResourceLimitExceeded")
        self.submitted.append((training, comparison_validation_s3_uri))
        name = job_name or f"job-{len(self.submitted)}"
        return SimpleNamespace(
            to_dict=lambda: {
                "job_name": name,
                "checkpoint_s3_uri": f"s3://checkpoints/recovery/{name}",
            },
            checkpoint_s3_uri=f"s3://checkpoints/recovery/{name}",
        )

    def status(self, job_name):
        return SimpleNamespace(status=self.job_status, to_dict=lambda: {"status": self.job_status})

    def _create_job(self, **request):
        self.test_request = request


def add_comparison_artifacts(service, items):
    for item in items:
        for key in ("validation_s3_uri", "best_model_s3_uri"):
            uri = item[key]
            service.s3.put_object(
                Bucket="checkpoints", Key=uri.split("checkpoints/", 1)[1], Body=b"{}"
            )
        prefix = item["checkpoint_s3_uri"].split("checkpoints/", 1)[1]
        service.s3.put_object(
            Bucket="checkpoints", Key=f"{prefix}/current.json",
            Body=b'{"version":1,"slot":"b","epoch":100}',
        )
        service.s3.put_object(
            Bucket="checkpoints", Key=f"{prefix}/resume-b.json",
            Body=json.dumps({"version": 1, "slot": "b", "epoch": 100,
                             "provider": service.config.training.provider}).encode(),
        )
        service.s3.put_object(
            Bucket="checkpoints", Key=f"{prefix}/resume-b.pt", Body=b"last weights"
        )


def test_comparison_submits_one_scratch_job_at_a_time():
    service = FakeService()
    models = ("yolox-m", "yolox-drax-mobilenet-v3-large-m-pyramid-drax")
    record = SubmitDetectionComparison(service, models).execute()

    assert [request.model for request, _ in service.submitted] == [models[0]]
    assert record.models[models[1]]["status"] == "Pending"
    assert AdvanceDetectionComparison(service, record.experiment_id).execute() is False
    service.job_status = "Completed"
    assert AdvanceDetectionComparison(service, record.experiment_id).execute() is True
    assert [request.model for request, _ in service.submitted] == list(models)
    assert all(not request.pretrained for request, _ in service.submitted)
    assert all(uri.endswith(f"{model}.json") for (request, uri), model in zip(service.submitted, models))
    assert set(record.models) == set(models)
    assert GetDetectionComparisonStatus(service, record.experiment_id).execute().models[models[0]]["status"] == "Completed"


def test_comparison_test_waits_for_both_jobs_and_uses_held_out_channels():
    service = FakeService()
    record = SubmitDetectionComparison(service, ("yolox-m", "yolox-drax-mobilenet-v3-large-m-pyramid-drax")).execute()
    launch = LaunchDetectionComparisonTest(service, record.experiment_id)
    with pytest.raises(MLXUserError, match="must be submitted"):
        launch.execute()
    service.job_status = "Completed"
    AdvanceDetectionComparison(service, record.experiment_id).execute()
    record = SubmitDetectionComparison(
        service, ("yolox-m", "yolox-drax-mobilenet-v3-large-m-pyramid-drax"),
        experiment_id=record.experiment_id,
    ).execute()
    add_comparison_artifacts(service, record.models.values())
    result = launch.execute()
    request = service.test_request
    assert result.test_job_name
    assert request["EnableManagedSpotTraining"] is False
    assert [channel["ChannelName"] for channel in request["InputDataConfig"]] == [
        "training", "model0", "model1", "model2", "model3"
    ]
    settings = json.loads(request["HyperParameters"]["mlx_comparison_test"])
    assert settings["models"] == {
        "yolox-m-best": "model0", "yolox-m-last": "model1",
        "yolox-drax-mobilenet-v3-large-m-pyramid-drax-best": "model2",
        "yolox-drax-mobilenet-v3-large-m-pyramid-drax-last": "model3",
    }
    assert list(settings["validation_s3_uris"]) == [
        "yolox-m-best", "yolox-drax-mobilenet-v3-large-m-pyramid-drax-best"
    ]
    assert request["InputDataConfig"][2]["DataSource"]["S3DataSource"]["S3Uri"].endswith("/resume-b.pt")
    assert settings["results_s3_uri"].endswith("/results.json")


def test_comparison_uses_configured_pretraining_and_spot():
    for options in ({"pretrained": True}, {"managed_spot": True}):
        service = FakeService(**options)
        SubmitDetectionComparison(service, ("yolox-m", "yolox-l")).execute()
        assert all(request.pretrained == options.get("pretrained", False) for request, _ in service.submitted)


def test_comparison_models_come_from_selected_mlx_provider(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("comparison:\n  models: [yolox-m, yolox-l]\n", encoding="utf-8")
    assert comparison_models(str(config), provider="libreyolo") == ("yolox-m", "yolox-l")
    assert comparison_models(
        str(config), provider="libreyolo", selected="yolox-m,yolox-drax-mobilenet-v3-large-m-pyramid-drax"
    )[0] == "yolox-m"
    with pytest.raises(MLXUserError, match="Unknown libreyolo comparison model"):
        comparison_models(str(config), provider="libreyolo", selected="yolox-m,invalid")


def test_comparison_config_accepts_cli_models_without_training_model(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "aws:\n  dataset_s3_uri: s3://data/quick.zip\n"
        "  checkpoint_s3_uri: s3://checkpoints\n"
        "  instance_type: ml.g4dn.2xlarge\n"
        "training:\n  provider: libreyolo\n  validate_after_training: true\n",
        encoding="utf-8",
    )
    loaded = load_aws_training_config(
        str(config), {"action": "compare-models", "models": "yolox-m,yolox-l"}
    )
    assert loaded.training.model == "yolox-m"
    assert comparison_models(str(config), provider=loaded.training.provider, selected="yolox-m,yolox-l") == ("yolox-m", "yolox-l")
    status_config = load_aws_training_config(str(config), {"action": "comparison-status"})
    assert status_config.training.provider == "libreyolo"


def test_comparison_stages_every_selected_model_for_test():
    service = FakeService(provider="ultralytics")
    models = ("yolo26", "draxnet-ave-yolo26", "draxnet-sknet-yolo26")
    record = SubmitDetectionComparison(service, models).execute()
    service.job_status = "Completed"
    AdvanceDetectionComparison(service, record.experiment_id).execute()
    AdvanceDetectionComparison(service, record.experiment_id).execute()
    record = SubmitDetectionComparison(service, models, experiment_id=record.experiment_id).execute()
    add_comparison_artifacts(service, record.models.values())
    LaunchDetectionComparisonTest(service, record.experiment_id).execute()
    channels = service.test_request["InputDataConfig"]
    assert [channel["ChannelName"] for channel in channels] == [
        "training", "model0", "model1", "model2", "model3", "model4", "model5"
    ]
    settings = json.loads(service.test_request["HyperParameters"]["mlx_comparison_test"])
    assert settings["provider"] == "ultralytics"


def test_comparison_rejects_a_stale_last_checkpoint():
    service = FakeService()
    record = SubmitDetectionComparison(service, ("yolox-m", "yolox-l")).execute()
    service.job_status = "Completed"
    AdvanceDetectionComparison(service, record.experiment_id).execute()
    record = SubmitDetectionComparison(
        service, ("yolox-m", "yolox-l"), experiment_id=record.experiment_id,
    ).execute()
    add_comparison_artifacts(service, record.models.values())
    first = record.models["yolox-m"]
    prefix = first["checkpoint_s3_uri"].split("checkpoints/", 1)[1]
    service.s3.put_object(
        Bucket="checkpoints", Key=f"{prefix}/current.json",
        Body=b'{"version":1,"slot":"b","epoch":99}',
    )
    with pytest.raises(MLXUserError, match="last checkpoint reached epoch 99, expected 100"):
        LaunchDetectionComparisonTest(service, record.experiment_id).execute()
    assert service.test_request is None


def test_watch_advances_pending_model_after_first_completes():
    service = FakeService()
    record = SubmitDetectionComparison(service, ("yolox-m", "yolox-l")).execute()
    status = GetDetectionComparisonStatus(service, record.experiment_id)
    observed = []

    def on_status(value):
        observed.append(value)
        service.job_status = "Completed"

    result = WatchDetectionComparison(
        status, interval=0.001, on_status=on_status,
        advance=AdvanceDetectionComparison(service, record.experiment_id),
    ).execute()
    assert len(observed) >= 2
    assert [request.model for request, _ in service.submitted] == ["yolox-m", "yolox-l"]
    assert all(item["status"] == "Completed" for item in result.models.values())


def test_watch_retries_quota_release_before_submitting_next_model():
    service = FakeService()
    record = SubmitDetectionComparison(service, ("yolox-m", "yolox-l")).execute()
    service.job_status = "Completed"
    service.fail_next_submit_quota = True
    result = WatchDetectionComparison(
        GetDetectionComparisonStatus(service, record.experiment_id),
        interval=0.001,
        advance=AdvanceDetectionComparison(service, record.experiment_id),
    ).execute()
    assert len(service.submitted) == 2
    assert result.models["yolox-l"]["status"] == "Completed"


def test_existing_partial_comparison_can_be_adopted_without_restarting_first_job():
    service = FakeService()
    models = ("yolox-m", "yolox-l")
    record = SubmitDetectionComparison(service, models).execute()
    bucket, key = "checkpoints", f"quick/comparisons/{record.experiment_id}/manifest.json"
    old = json.loads(service.s3.objects[(bucket, key)])
    old.pop("planned_models")
    old.pop("training_template")
    old["version"] = 1
    service.s3.objects[(bucket, key)] = json.dumps(old).encode()

    resumed = SubmitDetectionComparison(
        service, models, experiment_id=record.experiment_id
    ).execute()
    assert resumed.experiment_id == record.experiment_id
    assert len(service.submitted) == 1
    assert resumed.models["yolox-l"]["status"] == "Pending"


def test_sagemaker_comparison_entrypoint_benchmarks_test_split(tmp_path):
    data = tmp_path / "input" / "data"
    training = data / "training"
    training.mkdir(parents=True)
    with zipfile.ZipFile(training / "sample.zip", "w") as archive:
        archive.writestr("sample/data.yaml", "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {0: object}\n")
        for split in ("train", "val", "test"):
            archive.writestr(f"sample/images/{split}/.keep", "")
            archive.writestr(f"sample/labels/{split}/.keep", "")
    for channel in ("model0", "model1", "model2", "model3"):
        (data / channel).mkdir()
        filename = "best.pt" if channel in ("model0", "model2") else "resume-b.pt"
        (data / channel / filename).write_bytes(b"weights")
    requests = []

    class FakeBenchmark:
        def __init__(self, request):
            requests.append(request)

        def execute(self):
            return SimpleNamespace(metrics={"map50_95": 0.5})

    settings = {
        "models": {
            "yolox-m-best": "model0", "yolox-m-last": "model1",
            "drax-best": "model2", "drax-last": "model3",
        },
        "provider": "libreyolo",
        "height": 640, "width": 640, "batch_size": 8,
        "confidence": 0.001, "iou": 0.6, "max_detections": 300,
        "results_s3_uri": "s3://checkpoints/results.json",
        "results_csv_s3_uri": "s3://checkpoints/results.csv",
        "validation_s3_uris": {
            "yolox-m-best": "s3://checkpoints/val-m.json",
            "drax-best": "s3://checkpoints/val-drax.json",
        },
    }
    s3 = FakeS3()
    s3.put_object(Bucket="checkpoints", Key="val-m.json", Body=b'{"metrics":{"map_50_95":0.3}}')
    s3.put_object(Bucket="checkpoints", Key="val-drax.json", Body=b'{"metrics":{"map_50_95":0.4}}')
    result = RunSageMakerDetectionComparisonTest(
        hyperparameters={"mlx_comparison_test": json.dumps(settings)},
        input_dir=training, dataset_dir=tmp_path / "dataset",
        model_dir=tmp_path / "output", s3=s3, benchmark_command=FakeBenchmark,
    ).execute()
    assert [request.split for request in requests] == ["test"] * 4
    assert result["models"]["drax-last"]["test_metrics"]["map50_95"] == 0.5
    assert result["models"]["drax-last"]["checkpoint_file"] == "resume-b.pt"
    assert result["models"]["drax-last"]["validation_metrics"] is None
    assert result["models"]["drax-best"]["validation_metrics"]["map_50_95"] == 0.4
    assert ("checkpoints", "results.json") in s3.objects
    assert ("checkpoints", "results.csv") in s3.objects
