from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote, urlparse
from uuid import uuid4

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.aws.clients import AwsClientBundle
from mlx.modes.object_detection.aws.config import AwsTrainingConfig
from mlx.modes.object_detection.aws.image import PublishSageMakerImage, source_digest
from mlx.modes.object_detection.aws.models import (
    AwsInfrastructure,
    AwsTrainingStatus,
    AwsTrainingStopResult,
    AwsTrainingSubmission,
)
from mlx.modes.object_detection.aws.status import build_training_status, total_epochs
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _join_s3(uri: str, *parts: str) -> str:
    return "/".join([uri.rstrip("/"), *(part.strip("/") for part in parts)])


def _safe_job_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    return normalized or "mlx-od"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dockerfile() -> Path:
    return Path(__file__).with_name("Dockerfile")


class SageMakerTrainingService:
    """AWS integration boundary used by the object-detection AWS commands."""

    def __init__(
        self,
        config: AwsTrainingConfig,
        *,
        clients: AwsClientBundle | None = None,
    ) -> None:
        self.config = config
        resolved = clients or AwsClientBundle.create(config)
        self.session = resolved.session
        self.region = resolved.region
        self.s3 = resolved.s3
        self.ecr = resolved.ecr
        self.iam = resolved.iam
        self.sts = resolved.sts
        self.sagemaker = resolved.sagemaker
        self.cloudwatch = resolved.cloudwatch
        self._client_error = resolved.client_error
        self._boto_error = resolved.boto_error

    def _translate(self, action: str, exc: Exception) -> MLXUserError:
        return MLXUserError(f"AWS {action} failed: {exc}")

    def validate_storage(self) -> None:
        dataset_bucket, dataset_key = _parse_s3_uri(self.config.dataset_s3_uri)
        checkpoint_bucket, _ = _parse_s3_uri(self.config.checkpoint_s3_uri)
        try:
            self.s3.head_object(Bucket=dataset_bucket, Key=dataset_key)
            self.s3.head_bucket(Bucket=checkpoint_bucket)
            dataset_region = self._bucket_region(dataset_bucket)
            checkpoint_region = self._bucket_region(checkpoint_bucket)
        except (self._client_error, self._boto_error) as exc:
            raise self._translate("S3 validation", exc) from exc
        for label, bucket_region in (
            ("dataset", dataset_region),
            ("checkpoint", checkpoint_region),
        ):
            if bucket_region != self.region:
                raise MLXUserError(
                    f"The {label} S3 location is in region '{bucket_region}', but the "
                    f"SageMaker job is configured for '{self.region}'."
                )

    def _bucket_region(self, bucket: str) -> str:
        response = self.s3.get_bucket_location(Bucket=bucket)
        return response.get("LocationConstraint") or "us-east-1"

    def prepare_infrastructure(self) -> AwsInfrastructure:
        self.validate_storage()
        try:
            account_id = self.sts.get_caller_identity()["Account"]
            if self.config.image_uri:
                image_uri = self.config.image_uri
                repository_arn = (
                    self._repository_arn_from_image(image_uri)
                    if not self.config.execution_role_arn
                    else ""
                )
            else:
                repository_name, repository_uri, repository_arn = self._ensure_repository()
                image_uri = self._ensure_image(repository_name, repository_uri)
            role_arn = self.config.execution_role_arn or self._ensure_execution_role(
                repository_arn=repository_arn,
            )
        except MLXUserError:
            raise
        except (self._client_error, self._boto_error) as exc:
            raise self._translate("infrastructure preparation", exc) from exc
        return AwsInfrastructure(
            region=self.region,
            account_id=str(account_id),
            role_arn=role_arn,
            image_uri=image_uri,
        )

    def _repository_arn_from_image(self, image_uri: str) -> str:
        match = re.match(
            r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
            r"(?P<repository>[^:@]+)(?::[^@]+|@sha256:[a-f0-9]+)?$",
            image_uri,
        )
        if match is None:
            raise MLXUserError(
                "aws.image_uri must be an Amazon ECR image when MLX creates the execution role. "
                "Otherwise provide aws.execution_role_arn as well."
            )
        return (
            f"arn:aws:ecr:{match.group('region')}:{match.group('account')}:"
            f"repository/{match.group('repository')}"
        )

    def _ensure_repository(self) -> tuple[str, str, str]:
        name = self.config.ecr_repository or f"{self.config.resource_prefix}-training"
        try:
            repository = self.ecr.describe_repositories(repositoryNames=[name])["repositories"][0]
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") != "RepositoryNotFoundException":
                raise
            repository = self.ecr.create_repository(
                repositoryName=name,
                imageTagMutability="IMMUTABLE",
                imageScanningConfiguration={"scanOnPush": True},
                tags=[{"Key": "mlx:managed", "Value": "true"}],
            )["repository"]
        return name, repository["repositoryUri"], repository["repositoryArn"]

    def _source_digest(self) -> str:
        return source_digest(_package_root(), _dockerfile())

    def _ensure_image(self, repository_name: str, repository_uri: str) -> str:
        return PublishSageMakerImage(
            ecr=self.ecr,
            repository_name=repository_name,
            repository_uri=repository_uri,
            package_root=_package_root(),
            dockerfile=_dockerfile(),
            rebuild=self.config.rebuild_image,
            client_error=self._client_error,
            boto_error=self._boto_error,
        ).execute()

    def _ensure_execution_role(self, *, repository_arn: str) -> str:
        scope_digest = hashlib.sha256(
            "\n".join(
                (
                    self.config.dataset_s3_uri,
                    self.config.checkpoint_s3_uri,
                    repository_arn,
                )
            ).encode("utf-8")
        ).hexdigest()[:8]
        role_name = self.config.execution_role_name or (
            f"{self.config.resource_prefix}-sagemaker-{scope_digest}"
        )
        trust = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        }
        try:
            role = self.iam.get_role(RoleName=role_name)["Role"]
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
                raise
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust),
                Description="MLX SageMaker object-detection training execution role",
                Tags=[{"Key": "mlx:managed", "Value": "true"}],
            )["Role"]

        dataset_bucket, dataset_key = _parse_s3_uri(self.config.dataset_s3_uri)
        checkpoint_bucket, checkpoint_key = _parse_s3_uri(self.config.checkpoint_s3_uri)
        checkpoint_prefix = checkpoint_key.rstrip("/")
        checkpoint_object_arn = (
            f"arn:aws:s3:::{checkpoint_bucket}/{checkpoint_prefix}/*"
            if checkpoint_prefix
            else f"arn:aws:s3:::{checkpoint_bucket}/*"
        )
        checkpoint_list_prefix = f"{checkpoint_prefix}/*" if checkpoint_prefix else "*"
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ReadDataset",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{dataset_bucket}/{dataset_key}"],
                },
                {
                    "Sid": "InspectTrainingBuckets",
                    "Effect": "Allow",
                    "Action": ["s3:GetBucketLocation"],
                    "Resource": [
                        f"arn:aws:s3:::{dataset_bucket}",
                        f"arn:aws:s3:::{checkpoint_bucket}",
                    ],
                },
                {
                    "Sid": "ListDataset",
                    "Effect": "Allow",
                    "Action": ["s3:ListBucket"],
                    "Resource": [f"arn:aws:s3:::{dataset_bucket}"],
                    "Condition": {"StringLike": {"s3:prefix": [dataset_key]}},
                },
                {
                    "Sid": "ListTrainingPrefixes",
                    "Effect": "Allow",
                    "Action": ["s3:ListBucket"],
                    "Resource": [f"arn:aws:s3:::{checkpoint_bucket}"],
                    "Condition": {
                        "StringLike": {
                            "s3:prefix": [checkpoint_list_prefix]
                        }
                    },
                },
                {
                    "Sid": "ManageTrainingArtifacts",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
                    "Resource": [checkpoint_object_arn],
                },
                {
                    "Sid": "PullTrainingImage",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    "Resource": [repository_arn],
                },
                {
                    "Sid": "EcrAuthorization",
                    "Effect": "Allow",
                    "Action": ["ecr:GetAuthorizationToken"],
                    "Resource": ["*"],
                },
                {
                    "Sid": "TrainingLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                        "cloudwatch:PutMetricData",
                    ],
                    "Resource": ["*"],
                },
            ],
        }
        if self.config.kms_key_arn:
            policy["Statement"].append(
                {
                    "Sid": "TrainingKmsKey",
                    "Effect": "Allow",
                    "Action": [
                        "kms:Decrypt",
                        "kms:DescribeKey",
                        "kms:Encrypt",
                        "kms:GenerateDataKey",
                    ],
                    "Resource": [self.config.kms_key_arn],
                }
            )
        if self.config.vpc.subnet_ids or self.config.vpc.security_group_ids:
            policy["Statement"].append(
                {
                    "Sid": "TrainingVpcNetworkInterfaces",
                    "Effect": "Allow",
                    "Action": [
                        "ec2:CreateNetworkInterface",
                        "ec2:CreateNetworkInterfacePermission",
                        "ec2:DeleteNetworkInterface",
                        "ec2:DeleteNetworkInterfacePermission",
                        "ec2:DescribeDhcpOptions",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DescribeSecurityGroups",
                        "ec2:DescribeSubnets",
                        "ec2:DescribeVpcs",
                    ],
                    "Resource": ["*"],
                }
            )
        self.iam.put_role_policy(
            RoleName=role_name,
            PolicyName="MLXSageMakerTraining",
            PolicyDocument=json.dumps(policy),
        )
        return role["Arn"]

    def submit(
        self,
        infrastructure: AwsInfrastructure,
        *,
        run_id: Optional[str] = None,
        training: Optional[TrainObjectDetectionRequest] = None,
        training_payload: Optional[Mapping[str, Any]] = None,
        image_uri: Optional[str] = None,
        role_arn: Optional[str] = None,
        resume: bool = False,
    ) -> AwsTrainingSubmission:
        run_id = run_id or uuid4().hex
        request = training or self.config.training
        request_config = request.to_config()
        if training_payload is not None:
            serialized_request = TrainObjectDetectionRequest.from_config(training_payload)
            if serialized_request != request:
                raise MLXUserError(
                    "The serialized SageMaker training payload does not match the training request."
                )
            request_config = dict(training_payload)
        job_name = self._new_job_name(run_id)
        run_base = _join_s3(
            self.config.checkpoint_s3_uri,
            self.config.resource_prefix,
            "runs",
            run_id,
        )
        checkpoint_uri = _join_s3(run_base, "recovery")
        output_uri = _join_s3(run_base, "attempts", job_name)
        run_spec_uri = _join_s3(run_base, "run-spec.json")
        effective_image = image_uri or infrastructure.image_uri
        effective_role = role_arn or infrastructure.role_arn
        run_spec = {
            "version": 1,
            "run_id": run_id,
            "run_base_s3_uri": run_base,
            "dataset_s3_uri": self.config.dataset_s3_uri,
            "checkpoint_base_s3_uri": self.config.checkpoint_s3_uri,
            "resource_prefix": self.config.resource_prefix,
            "image_uri": effective_image,
            "role_arn": effective_role,
            "training": request_config,
        }
        if not resume:
            self._put_json(run_spec_uri, run_spec)
        attempt_spec_uri = _join_s3(run_base, "attempts", job_name, "attempt-spec.json")
        self._put_json(
            attempt_spec_uri,
            {**run_spec, "job_name": job_name, "resume": resume},
        )

        algorithm = {
            "TrainingImage": effective_image,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
            "MetricDefinitions": [
                {"Name": "mlx:epoch", "Regex": r"MLX_EPOCH=([0-9.]+);"},
                {"Name": "mlx:progress", "Regex": r"MLX_PROGRESS=([0-9.]+);"},
                {"Name": "mlx:eta_seconds", "Regex": r"MLX_ETA_SECONDS=([0-9.]+);"},
            ],
        }
        stopping = {"MaxRuntimeInSeconds": self.config.max_runtime_seconds}
        if self.config.managed_spot:
            stopping["MaxWaitTimeInSeconds"] = self.config.effective_max_wait_seconds

        create_request: dict[str, Any] = {
            "TrainingJobName": job_name,
            "RoleArn": effective_role,
            "AlgorithmSpecification": algorithm,
            "InputDataConfig": [{
                "ChannelName": "training",
                "ContentType": "application/zip",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": self.config.dataset_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }],
            "OutputDataConfig": {"S3OutputPath": output_uri},
            "ResourceConfig": {
                "InstanceType": self.config.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": self.config.volume_size_gb,
            },
            "StoppingCondition": stopping,
            "EnableManagedSpotTraining": self.config.managed_spot,
            "EnableNetworkIsolation": self.config.network_isolation,
            "CheckpointConfig": {
                "S3Uri": checkpoint_uri,
                "LocalPath": "/opt/ml/checkpoints",
            },
            "HyperParameters": {
                "mlx_run_id": run_id,
                "mlx_run_spec_s3_uri": run_spec_uri,
                "mlx_attempt_spec_s3_uri": attempt_spec_uri,
                "mlx_training": json.dumps(request_config, separators=(",", ":")),
                "mlx_resume": str(resume).lower(),
                "mlx_image_uri": effective_image,
                "mlx_volume_size_gb": str(self.config.volume_size_gb),
            },
            "Tags": [
                {"Key": "mlx:managed", "Value": "true"},
                {"Key": "mlx:run-id", "Value": run_id},
                *({"Key": str(key), "Value": str(value)} for key, value in self.config.tags.items()),
            ],
        }
        if self.config.kms_key_arn:
            create_request["OutputDataConfig"]["KmsKeyId"] = self.config.kms_key_arn
        if self.config.vpc.subnet_ids or self.config.vpc.security_group_ids:
            if not self.config.vpc.subnet_ids or not self.config.vpc.security_group_ids:
                raise MLXUserError(
                    "Both aws.vpc.subnet_ids and security_group_ids are required when VPC training is enabled."
                )
            create_request["VpcConfig"] = {
                "Subnets": list(self.config.vpc.subnet_ids),
                "SecurityGroupIds": list(self.config.vpc.security_group_ids),
            }

        response = self._create_training_job(create_request)
        return AwsTrainingSubmission(
            job_name=job_name,
            job_arn=response["TrainingJobArn"],
            run_id=run_id,
            status="InProgress",
            region=self.region,
            managed_spot=self.config.managed_spot,
            image_uri=effective_image,
            checkpoint_s3_uri=checkpoint_uri,
            output_s3_uri=output_uri,
            console_url=self._console_url(job_name),
        )

    def resume(self, old_job_name: str) -> AwsTrainingSubmission:
        old = self._describe(old_job_name)
        if old["TrainingJobStatus"] not in {"Stopped", "Failed", "Completed"}:
            raise MLXUserError(
                f"Training job '{old_job_name}' must be terminal before it can be resumed."
            )
        hyperparameters = old.get("HyperParameters", {})
        run_id = hyperparameters.get("mlx_run_id")
        run_spec_uri = hyperparameters.get("mlx_run_spec_s3_uri")
        if not run_id or not run_spec_uri:
            raise MLXUserError(f"Training job '{old_job_name}' is not an MLX recoverable job.")
        self._ensure_no_active_attempt(run_id, excluding=old_job_name)
        spec = self._get_json(run_spec_uri)
        if spec.get("dataset_s3_uri") != self.config.dataset_s3_uri:
            raise MLXUserError("The resume configuration must use the original dataset_s3_uri.")
        if spec.get("checkpoint_base_s3_uri") != self.config.checkpoint_s3_uri:
            raise MLXUserError("The resume configuration must use the original checkpoint_s3_uri.")
        original_payload = spec.get("training")
        if not isinstance(original_payload, Mapping):
            raise MLXUserError(
                f"Training job '{old_job_name}' has an invalid MLX run specification."
            )
        original = TrainObjectDetectionRequest.from_config(original_payload)
        current = self.config.training
        if (current.provider, current.model) != (original.provider, original.model):
            raise MLXUserError("Provider and model cannot change when resuming a training run.")
        if current.epochs < original.epochs:
            raise MLXUserError("A resumed total epoch target cannot be lower than the original target.")
        completed_epoch = self._latest_recovery_epoch(spec["run_base_s3_uri"], original.provider)
        if current.epochs <= completed_epoch:
            raise MLXUserError(
                f"The run already has a recoverable checkpoint at epoch {completed_epoch}; "
                "set a higher total epoch target to continue it."
            )
        if old["TrainingJobStatus"] == "Completed" and current.epochs <= original.epochs:
            raise MLXUserError(
                "A completed run can only be resumed with a higher total epoch target."
            )
        resumed_payload = dict(original_payload)
        resumed_payload["epochs"] = current.epochs
        resumed = TrainObjectDetectionRequest.from_config(resumed_payload)
        infrastructure = AwsInfrastructure(
            region=self.region,
            account_id=old["TrainingJobArn"].split(":")[4],
            role_arn=spec["role_arn"],
            image_uri=spec["image_uri"],
        )
        return self.submit(
            infrastructure,
            run_id=run_id,
            training=resumed,
            training_payload=resumed_payload,
            image_uri=spec["image_uri"],
            role_arn=spec["role_arn"],
            resume=True,
        )

    def _latest_recovery_epoch(self, run_base_uri: str, provider: str) -> int:
        recovery_uri = _join_s3(run_base_uri, "recovery")
        epochs: list[int] = []
        for slot in ("a", "b"):
            metadata_uri = _join_s3(recovery_uri, f"resume-{slot}.json")
            checkpoint_uri = _join_s3(recovery_uri, f"resume-{slot}.pt")
            try:
                metadata = self._get_optional_json(metadata_uri)
                if metadata is None:
                    continue
                bucket, key = _parse_s3_uri(checkpoint_uri)
                self.s3.head_object(Bucket=bucket, Key=key)
                if metadata.get("provider") == provider:
                    epochs.append(int(metadata["epoch"]))
            except self._client_error as exc:
                if exc.response.get("Error", {}).get("Code") in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    continue
                raise self._translate("recovery checkpoint lookup", exc) from exc
            except self._boto_error as exc:
                raise self._translate("recovery checkpoint lookup", exc) from exc
            except (KeyError, TypeError, ValueError):
                continue
        if not epochs:
            raise MLXUserError(
                f"No recoverable {provider} checkpoint was found under {recovery_uri}."
            )
        return max(epochs)

    def _create_training_job(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        delays = (0, 2, 4, 8, 16)
        for index, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                return self.sagemaker.create_training_job(**request)
            except self._client_error as exc:
                error = exc.response.get("Error", {})
                message = str(error.get("Message", "")).lower()
                role_propagation = (
                    error.get("Code") == "ValidationException"
                    and ("role" in message or "assume" in message)
                )
                if not role_propagation or index == len(delays) - 1:
                    raise self._translate("training job submission", exc) from exc
            except self._boto_error as exc:
                raise self._translate("training job submission", exc) from exc
        raise MLXUserError("AWS training job submission failed after IAM propagation retries.")

    def _ensure_no_active_attempt(self, run_id: str, *, excluding: str) -> None:
        for status in ("InProgress", "Stopping"):
            try:
                paginator = self.sagemaker.get_paginator("list_training_jobs")
                pages = paginator.paginate(
                    StatusEquals=status,
                    NameContains=_safe_job_component(self.config.resource_prefix)[:32],
                )
                for page in pages:
                    for summary in page.get("TrainingJobSummaries", []):
                        candidate = summary["TrainingJobName"]
                        if candidate == excluding:
                            continue
                        description = self._describe(candidate)
                        if description.get("HyperParameters", {}).get("mlx_run_id") == run_id:
                            raise MLXUserError(
                                f"Logical run '{run_id}' already has active SageMaker job "
                                f"'{candidate}'."
                            )
            except (self._client_error, self._boto_error) as exc:
                raise self._translate("active training attempt lookup", exc) from exc

    def status(self, job_name: str) -> AwsTrainingStatus:
        description = self._describe(job_name)
        return build_training_status(
            description,
            latest_metric=self._latest_metric,
            console_url=self._console_url(job_name),
        )

    def stop(self, job_name: str, *, config_path: str) -> AwsTrainingStopResult:
        current = self.status(job_name)
        if not current.terminal:
            try:
                self.sagemaker.stop_training_job(TrainingJobName=job_name)
            except (self._client_error, self._boto_error) as exc:
                raise self._translate("training job stop", exc) from exc
            status = "Stopping"
        else:
            status = current.status
        command = (
            "python -m mlx --mode object-detection --platform aws --action resume "
            f"--config {config_path} --job-name {job_name}"
        )
        return AwsTrainingStopResult(
            job_name=job_name,
            status=status,
            checkpoint_s3_uri=current.checkpoint_s3_uri,
            resume_command=command,
        )

    def _new_job_name(self, run_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix = _safe_job_component(self.config.resource_prefix)[:32]
        return f"{prefix}-{timestamp}-{run_id[:8]}-{uuid4().hex[:4]}"[:63].rstrip("-")

    def _put_json(self, uri: str, value: Mapping[str, Any]) -> None:
        bucket, key = _parse_s3_uri(uri)
        try:
            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(value, sort_keys=True).encode("utf-8"),
                ContentType="application/json",
            )
        except (self._client_error, self._boto_error) as exc:
            raise self._translate("run manifest upload", exc) from exc

    def _get_json(self, uri: str) -> Mapping[str, Any]:
        bucket, key = _parse_s3_uri(uri)
        try:
            body = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            value = json.loads(body)
        except (self._client_error, self._boto_error, json.JSONDecodeError) as exc:
            raise self._translate("run manifest retrieval", exc) from exc
        if not isinstance(value, Mapping):
            raise MLXUserError(f"AWS run manifest is invalid: {uri}")
        return value

    def _get_optional_json(self, uri: str) -> Optional[Mapping[str, Any]]:
        bucket, key = _parse_s3_uri(uri)
        try:
            body = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return None
            raise self._translate("recovery metadata retrieval", exc) from exc
        except self._boto_error as exc:
            raise self._translate("recovery metadata retrieval", exc) from exc
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, Mapping) else None

    def _describe(self, job_name: str) -> Mapping[str, Any]:
        try:
            return self.sagemaker.describe_training_job(TrainingJobName=job_name)
        except (self._client_error, self._boto_error) as exc:
            raise self._translate(f"training job lookup for '{job_name}'", exc) from exc

    def _latest_metric(self, description: Mapping[str, Any], metric_name: str) -> Optional[float]:
        creation = description.get("CreationTime")
        if creation is None:
            return None
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace="/aws/sagemaker/TrainingJobs",
                MetricName=metric_name,
                Dimensions=[{
                    "Name": "TrainingJobName",
                    "Value": description["TrainingJobName"],
                }],
                StartTime=creation,
                EndTime=datetime.now(timezone.utc),
                Period=60,
                Statistics=["Maximum"],
            )
        except (self._client_error, self._boto_error):
            return None
        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return None
        return float(max(datapoints, key=lambda item: item["Timestamp"])["Maximum"])

    @staticmethod
    def _total_epochs(hyperparameters: Mapping[str, str]) -> Optional[int]:
        return total_epochs(hyperparameters)

    def _console_url(self, job_name: str) -> str:
        return (
            f"https://{self.region}.console.aws.amazon.com/sagemaker/home"
            f"?region={self.region}#/jobs/{quote(job_name)}"
        )
