"""SageMaker comparison of independently trained object detectors."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from uuid import UUID, uuid4, uuid5

from mlx.core.exceptions import MLXUserError
from mlx.core.aws.models import AwsInfrastructure
from mlx.modes.object_detection.aws.config import _load_yaml
from mlx.modes.object_detection.aws.service import SageMakerTrainingService
from mlx.modes.object_detection.providers import get_provider


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise MLXUserError(f"Invalid comparison S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


@dataclass(frozen=True)
class ComparisonRecord:
    experiment_id: str
    manifest_s3_uri: str
    models: Mapping[str, Any]
    test_job_name: str | None = None
    results_s3_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def comparison_models(
    config_path: str, *, provider: str, selected: str | None = None,
) -> tuple[str, ...]:
    root = _load_yaml(Path(config_path).expanduser())
    raw = root.get("comparison", {})
    if not isinstance(raw, Mapping) or set(raw) - {"models"}:
        raise MLXUserError("comparison must be a YAML mapping containing only models.")
    models = selected.split(",") if selected is not None else raw.get("models")
    if (not isinstance(models, list) or len(models) < 2
            or not all(isinstance(name, str) and name.strip() for name in models)):
        raise MLXUserError("Pass at least two model names with --models or comparison.models.")
    names = tuple(name.strip() for name in models)
    return validate_model_selection(names, provider=provider)


def validate_model_selection(names: tuple[str, ...], *, provider: str) -> tuple[str, ...]:
    if len(names) < 2 or not all(isinstance(name, str) and name.strip() for name in names):
        raise MLXUserError("Comparison requires at least two MLX model names.")
    if len(set(names)) != len(names):
        raise MLXUserError("Comparison model names must be distinct.")
    provider_adapter = get_provider(provider)
    catalog = getattr(provider_adapter, "model_names", None)
    if not callable(catalog):
        raise MLXUserError(f"Provider '{provider}' does not expose an MLX model catalog.")
    available = set(catalog())
    unknown = sorted(set(names) - available)
    if unknown:
        raise MLXUserError(
            f"Unknown {provider} comparison model(s): {', '.join(unknown)}. "
            f"Use --action ls-models --provider {provider} --names-only to list available names."
        )
    return names


class DetectionComparisonStore:
    def __init__(self, service: SageMakerTrainingService):
        self.service = service

    def prefix(self, experiment_id: str) -> str:
        if len(experiment_id) != 32 or any(c not in "0123456789abcdef" for c in experiment_id):
            raise MLXUserError("--experiment-id must be the 32-character ID returned at submission.")
        config = self.service.config
        return f"{config.checkpoint_s3_uri.rstrip('/')}/{config.resource_prefix}/comparisons/{experiment_id}"

    def read(self, experiment_id: str) -> dict[str, Any]:
        uri = f"{self.prefix(experiment_id)}/manifest.json"
        bucket, key = _s3_parts(uri)
        try:
            body = self.service.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            manifest = json.loads(body)
        except Exception as exc:
            raise MLXUserError(f"Unable to read comparison manifest at {uri}: {exc}") from exc
        if manifest.get("dataset_s3_uri") != self.service.config.dataset_s3_uri:
            raise MLXUserError("Comparison dataset differs from the supplied configuration.")
        return manifest

    def write(self, manifest: Mapping[str, Any]) -> None:
        uri = f"{self.prefix(manifest['experiment_id'])}/manifest.json"
        bucket, key = _s3_parts(uri)
        try:
            self.service.s3.put_object(
                Bucket=bucket, Key=key,
                Body=json.dumps(manifest, sort_keys=True).encode(),
                ContentType="application/json",
            )
        except Exception as exc:
            raise MLXUserError(f"Unable to write comparison manifest at {uri}: {exc}") from exc


class SubmitDetectionComparison:
    def __init__(
        self, service: SageMakerTrainingService, models: tuple[str, ...],
        *, experiment_id: str | None = None,
    ):
        self.service = service
        self.models = models
        self.experiment_id = experiment_id
        self.store = DetectionComparisonStore(service)

    def execute(self) -> ComparisonRecord:
        config = self.service.config
        validate_model_selection(self.models, provider=config.training.provider)
        if not config.training.validate_after_training or config.training.validation_split != "val":
            raise MLXUserError("Model comparison requires validation on the val split.")
        if self.experiment_id:
            manifest = self.store.read(self.experiment_id)
            existing = tuple(manifest.get("planned_models") or manifest["models"])
            if (existing != self.models if manifest.get("planned_models")
                    else existing != self.models[:len(existing)]):
                raise MLXUserError("Selected models do not match the existing comparison order.")
            if manifest.get("provider") != config.training.provider:
                raise MLXUserError("Comparison provider differs from the supplied configuration.")
            if (manifest.get("training_template") is not None
                    and manifest["training_template"] != config.training.to_config()):
                raise MLXUserError("Comparison training settings differ from the original submission.")
            manifest["planned_models"] = list(self.models)
            manifest.setdefault("training_template", config.training.to_config())
        else:
            infrastructure = self.service.prepare_infrastructure()
            manifest = {
                "version": 2,
                "experiment_id": uuid4().hex,
                "dataset_s3_uri": config.dataset_s3_uri,
                "provider": config.training.provider,
                "image_uri": infrastructure.image_uri,
                "role_arn": infrastructure.role_arn,
                "planned_models": list(self.models),
                "training_template": config.training.to_config(),
                "models": {},
            }
        self.store.write(manifest)
        AdvanceDetectionComparison(self.service, manifest["experiment_id"]).execute()
        manifest = self.store.read(manifest["experiment_id"])
        prefix = self.store.prefix(manifest["experiment_id"])
        return ComparisonRecord(
            manifest["experiment_id"], f"{prefix}/manifest.json",
            {
                name: manifest["models"].get(name, {"status": "Pending"})
                for name in manifest["planned_models"]
            },
        )


class AdvanceDetectionComparison:
    """Submit the next model only after all earlier jobs have completed."""

    def __init__(self, service: SageMakerTrainingService, experiment_id: str):
        self.service = service
        self.experiment_id = experiment_id
        self.store = DetectionComparisonStore(service)

    def execute(self) -> bool:
        manifest = self.store.read(self.experiment_id)
        planned = tuple(manifest.get("planned_models") or manifest["models"])
        submitted = manifest["models"]
        pending = [model for model in planned if model not in submitted]
        if not pending:
            return False
        for item in submitted.values():
            if self.service.status(item["job_name"]).status != "Completed":
                return False
        model = pending[0]
        index = planned.index(model)
        run_id = uuid5(UUID(self.experiment_id), model).hex
        job_name = (
            f"{self.service.config.resource_prefix[:30]}-cmp-{self.experiment_id[:8]}-{index}"
        )[:63]
        config = self.service.config
        request = replace(
            config.training,
            model=model,
            run_name=f"comparison-{self.experiment_id[:8]}-{model}"[:80],
        )
        prefix = self.store.prefix(self.experiment_id)
        validation_uri = f"{prefix}/validation/{model}.json"
        infrastructure = AwsInfrastructure(
            region=self.service.region,
            account_id=manifest["role_arn"].split(":")[4],
            role_arn=manifest["role_arn"],
            image_uri=manifest["image_uri"],
        )
        try:
            submission = self.service.submit(
                infrastructure,
                run_id=run_id,
                job_name=job_name,
                training=request,
                comparison_validation_s3_uri=validation_uri,
            )
            submission_data = submission.to_dict()
            checkpoint_uri = submission.checkpoint_s3_uri
        except MLXUserError as exc:
            try:
                description = self.service.sagemaker.describe_training_job(
                    TrainingJobName=job_name
                )
            except Exception:
                raise exc
            checkpoint_uri = (
                f"{config.checkpoint_s3_uri.rstrip('/')}/{config.resource_prefix}"
                f"/runs/{run_id}/recovery"
            )
            submission_data = {
                "job_name": job_name,
                "job_arn": description["TrainingJobArn"],
                "run_id": run_id,
                "status": description["TrainingJobStatus"],
                "checkpoint_s3_uri": checkpoint_uri,
            }
        submitted[model] = {
            **submission_data,
            "validation_s3_uri": validation_uri,
            "best_model_s3_uri": f"{checkpoint_uri}/best.pt",
        }
        self.store.write(manifest)
        return True


class GetDetectionComparisonStatus:
    def __init__(self, service: SageMakerTrainingService, experiment_id: str):
        self.service = service
        self.experiment_id = experiment_id
        self.store = DetectionComparisonStore(service)

    def execute(self) -> ComparisonRecord:
        manifest = self.store.read(self.experiment_id)
        statuses = {}
        for model in manifest.get("planned_models") or manifest["models"]:
            item = manifest["models"].get(model)
            if item is None:
                statuses[model] = {"status": "Pending"}
                continue
            status = self.service.status(item["job_name"]).to_dict()
            if status["status"] == "Completed":
                status["validation"] = self._read_optional_json(item["validation_s3_uri"])
            statuses[model] = status
        test_job = manifest.get("test_job_name")
        if test_job:
            statuses["test"] = self.service.status(test_job).to_dict()
            if statuses["test"]["status"] == "Completed":
                statuses["test"]["results"] = self._read_optional_json(
                    f"{self.store.prefix(self.experiment_id)}/results.json"
                )
        prefix = self.store.prefix(self.experiment_id)
        return ComparisonRecord(
            self.experiment_id,
            f"{prefix}/manifest.json",
            statuses,
            test_job,
            f"{prefix}/results.json" if test_job else None,
        )

    def _read_optional_json(self, uri: str) -> Any:
        bucket, key = _s3_parts(uri)
        try:
            body = self.service.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if isinstance(exc, KeyError) or code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise MLXUserError(f"Unable to read comparison result at {uri}: {exc}") from exc
        try:
            return json.loads(body)
        except (TypeError, ValueError) as exc:
            raise MLXUserError(f"Comparison result at {uri} is not valid JSON: {exc}") from exc


class WatchDetectionComparison:
    def __init__(
        self, status: GetDetectionComparisonStatus, *, interval: float = 30.0,
        on_status=None, advance: AdvanceDetectionComparison | None = None,
    ):
        self.status = status
        self.interval = interval
        self.on_status = on_status
        self.advance = advance

    def execute(self) -> ComparisonRecord:
        if self.interval <= 0:
            raise MLXUserError("--poll-interval must be greater than zero.")
        while True:
            if self.advance is not None:
                try:
                    self.advance.execute()
                except MLXUserError as exc:
                    if "ResourceLimitExceeded" not in str(exc):
                        raise
            record = self.status.execute()
            if self.on_status:
                self.on_status(record)
            states = [item["status"] for name, item in record.models.items() if name != "test"]
            if any(state in {"Stopped", "Failed"} for state in states):
                return record
            if states and all(state == "Completed" for state in states) and (
                not record.test_job_name or record.models["test"]["status"] in {"Completed", "Stopped", "Failed"}
            ):
                return record
            time.sleep(self.interval)


class LaunchDetectionComparisonTest:
    def __init__(self, service: SageMakerTrainingService, experiment_id: str):
        self.service = service
        self.experiment_id = experiment_id
        self.store = DetectionComparisonStore(service)

    def execute(self) -> ComparisonRecord:
        manifest = self.store.read(self.experiment_id)
        if manifest.get("test_job_name"):
            raise MLXUserError("This comparison already has a test job.")
        planned = tuple(manifest.get("planned_models") or manifest["models"])
        if len(planned) < 2 or len(manifest["models"]) != len(planned):
            raise MLXUserError("All selected model training jobs must be submitted before testing.")
        statuses = GetDetectionComparisonStatus(self.service, self.experiment_id).execute()
        if any(item["status"] != "Completed" for item in statuses.models.values()):
            raise MLXUserError("All training jobs must complete before testing.")
        prefix = self.store.prefix(self.experiment_id)
        inputs = [self._channel("training", manifest["dataset_s3_uri"])]
        model_channels = {}
        validation_uris = {}
        for name in planned:
            record = manifest["models"][name]
            validation_bucket, validation_key = _s3_parts(record["validation_s3_uri"])
            try:
                self.service.s3.head_object(Bucket=validation_bucket, Key=validation_key)
            except Exception as exc:
                raise MLXUserError(f"Validation metrics for {name} are unavailable: {exc}") from exc
            bucket, key = _s3_parts(record["best_model_s3_uri"])
            try:
                self.service.s3.head_object(Bucket=bucket, Key=key)
            except Exception as exc:
                raise MLXUserError(f"Best checkpoint for {name} is unavailable: {exc}") from exc
            last_uri = self._final_checkpoint_uri(name, record, manifest)
            for kind, uri in (("best", record["best_model_s3_uri"]), ("last", last_uri)):
                channel = f"model{len(model_channels)}"
                inputs.append(self._channel(channel, uri))
                model_channels[f"{name}-{kind}"] = channel
            validation_uris[f"{name}-best"] = record["validation_s3_uri"]
        job_name = f"{self.service.config.resource_prefix[:30]}-test-{self.experiment_id[:8]}"
        config = self.service.config
        request = {
            "TrainingJobName": job_name,
            "RoleArn": manifest["role_arn"],
            "AlgorithmSpecification": {
                "TrainingImage": manifest["image_uri"], "TrainingInputMode": "File"
            },
            "InputDataConfig": inputs,
            "OutputDataConfig": {"S3OutputPath": f"{prefix}/test"},
            "ResourceConfig": {
                "InstanceType": config.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": config.volume_size_gb,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": min(config.max_runtime_seconds, 7200)},
            "EnableManagedSpotTraining": False,
            "EnableNetworkIsolation": config.network_isolation,
            "HyperParameters": {
                "mlx_comparison_test": json.dumps({
                    "models": model_channels,
                    "provider": manifest["provider"],
                    "validation_s3_uris": validation_uris,
                    "height": config.training.height,
                    "width": config.training.width,
                    "batch_size": config.training.batch_size,
                    "confidence": config.training.validation_confidence,
                    "iou": config.training.validation_iou,
                    "max_detections": config.training.validation_max_detections,
                    "results_s3_uri": f"{prefix}/results.json",
                    "results_csv_s3_uri": f"{prefix}/results.csv",
                }),
                "mlx_volume_size_gb": str(config.volume_size_gb),
            },
        }
        if config.kms_key_arn:
            request["OutputDataConfig"]["KmsKeyId"] = config.kms_key_arn
        if config.vpc.subnet_ids or config.vpc.security_group_ids:
            request["VpcConfig"] = {
                "Subnets": list(config.vpc.subnet_ids),
                "SecurityGroupIds": list(config.vpc.security_group_ids),
            }
        try:
            self.service.sagemaker.create_training_job(**request)
        except Exception as exc:
            raise MLXUserError(f"Unable to start comparison test job: {exc}") from exc
        manifest["test_job_name"] = job_name
        self.store.write(manifest)
        return ComparisonRecord(
            self.experiment_id, f"{prefix}/manifest.json",
            manifest["models"], job_name, f"{prefix}/results.json"
        )

    def _final_checkpoint_uri(
        self, name: str, record: Mapping[str, Any], manifest: Mapping[str, Any],
    ) -> str:
        recovery_uri = record["checkpoint_s3_uri"].rstrip("/")
        bucket, key = _s3_parts(f"{recovery_uri}/current.json")
        try:
            current = json.loads(self.service.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            slot = current["slot"]
            epoch = current["epoch"]
            if current.get("version") != 1 or slot not in ("a", "b") or type(epoch) is not int:
                raise ValueError("invalid active recovery slot")
            expected_epochs = int(manifest.get("training_template", {}).get(
                "epochs", self.service.config.training.epochs,
            ))
            if epoch != expected_epochs:
                raise ValueError(f"last checkpoint reached epoch {epoch}, expected {expected_epochs}")
            metadata_key = f"{key.rsplit('/', 1)[0]}/resume-{slot}.json"
            metadata = json.loads(
                self.service.s3.get_object(Bucket=bucket, Key=metadata_key)["Body"].read()
            )
            if (metadata.get("version") != 1 or metadata.get("slot") != slot
                    or metadata.get("epoch") != epoch
                    or metadata.get("provider") != manifest["provider"]):
                raise ValueError("recovery metadata does not match the final checkpoint")
            checkpoint_uri = f"{recovery_uri}/resume-{slot}.pt"
            checkpoint_bucket, checkpoint_key = _s3_parts(checkpoint_uri)
            self.service.s3.head_object(Bucket=checkpoint_bucket, Key=checkpoint_key)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise MLXUserError(f"Final last.pt checkpoint for {name} is unavailable: {exc}") from exc
        except Exception as exc:
            raise MLXUserError(f"Unable to inspect final last.pt checkpoint for {name}: {exc}") from exc
        return checkpoint_uri

    @staticmethod
    def _channel(name: str, uri: str) -> dict[str, Any]:
        return {
            "ChannelName": name,
            "InputMode": "File",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": uri,
                "S3DataDistributionType": "FullyReplicated",
            }},
        }
