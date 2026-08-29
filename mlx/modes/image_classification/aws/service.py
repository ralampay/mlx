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
from mlx.modes.image_classification.aws.clients import AwsClientBundle
from mlx.modes.image_classification.aws.config import (
    AwsTrainingConfig,
    serialize_training_request,
)
from mlx.modes.image_classification.aws.image import PublishSageMakerImage
from mlx.modes.image_classification.aws.models import (
    AwsInfrastructure,
    AwsTrainingStatus,
    AwsTrainingStopResult,
    AwsTrainingSubmission,
)
from mlx.modes.image_classification.aws.status import build_training_status
from mlx.modes.image_classification.requests import ImageClassificationRequest


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _join_s3(uri: str, *parts: str) -> str:
    return "/".join([uri.rstrip("/"), *(part.strip("/") for part in parts)])


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-") or "mlx-ic"


class SageMakerTrainingService:
    """AWS integration boundary for image-classification training lifecycle operations."""

    def __init__(
        self,
        config: AwsTrainingConfig,
        *,
        clients: AwsClientBundle | None = None,
    ) -> None:
        self.config = config
        bundle = clients or AwsClientBundle.create(config)
        self.region = bundle.region
        self.s3 = bundle.s3
        self.ecr = bundle.ecr
        self.iam = bundle.iam
        self.sts = bundle.sts
        self.sagemaker = bundle.sagemaker
        self.cloudwatch = bundle.cloudwatch
        self._client_error = bundle.client_error
        self._boto_error = bundle.boto_error

    def _error(self, action: str, exc: Exception) -> MLXUserError:
        return MLXUserError(f"AWS {action} failed: {exc}")

    def validate_storage(self) -> None:
        dataset_bucket, dataset_key = _parse_s3(self.config.dataset_s3_uri)
        checkpoint_bucket, _ = _parse_s3(self.config.checkpoint_s3_uri)
        try:
            self.s3.head_object(Bucket=dataset_bucket, Key=dataset_key)
            self.s3.head_bucket(Bucket=checkpoint_bucket)
            regions = (
                ("dataset", self._bucket_region(dataset_bucket)),
                ("checkpoint", self._bucket_region(checkpoint_bucket)),
            )
        except (self._client_error, self._boto_error) as exc:
            raise self._error("S3 validation", exc) from exc
        for label, region in regions:
            if region != self.region:
                raise MLXUserError(
                    f"The {label} S3 location is in region '{region}', but the "
                    f"SageMaker job is configured for '{self.region}'."
                )

    def _bucket_region(self, bucket: str) -> str:
        return self.s3.get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"

    def prepare_infrastructure(self) -> AwsInfrastructure:
        self.validate_storage()
        try:
            account = str(self.sts.get_caller_identity()["Account"])
            if self.config.image_uri:
                image_uri = self.config.image_uri
                repository_arn = "" if self.config.execution_role_arn else self._repository_arn(image_uri)
            else:
                repository_name, repository_uri, repository_arn = self._ensure_repository()
                image_uri = PublishSageMakerImage(
                    ecr=self.ecr, repository_name=repository_name, repository_uri=repository_uri,
                    package_root=Path(__file__).resolve().parents[3],
                    dockerfile=Path(__file__).with_name("Dockerfile"),
                    rebuild=self.config.rebuild_image,
                    client_error=self._client_error,
                    boto_error=self._boto_error,
                ).execute()
            role_arn = self.config.execution_role_arn or self._ensure_execution_role(repository_arn)
        except MLXUserError:
            raise
        except (self._client_error, self._boto_error) as exc:
            raise self._error("infrastructure preparation", exc) from exc
        return AwsInfrastructure(self.region, account, role_arn, image_uri)

    def _repository_arn(self, image_uri: str) -> str:
        match = re.match(
            r"^(?P<account>[0-9]{12})\.dkr\.ecr\."
            r"(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
            r"(?P<repository>[^:@]+)(?::[^@]+|@sha256:[a-f0-9]+)?$",
            image_uri,
        )
        if not match:
            raise MLXUserError(
                "aws.image_uri must be an Amazon ECR image when MLX creates the "
                "execution role. Otherwise provide aws.execution_role_arn as well."
            )
        return f"arn:aws:ecr:{match['region']}:{match['account']}:repository/{match['repository']}"

    def _ensure_repository(self) -> tuple[str, str, str]:
        name = self.config.ecr_repository or f"{self.config.resource_prefix}-training"
        try:
            repository = self.ecr.describe_repositories(repositoryNames=[name])["repositories"][0]
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") != "RepositoryNotFoundException":
                raise
            repository = self.ecr.create_repository(
                repositoryName=name, imageTagMutability="IMMUTABLE",
                imageScanningConfiguration={"scanOnPush": True}, tags=[{"Key": "mlx:managed", "Value": "true"}],
            )["repository"]
        return name, repository["repositoryUri"], repository["repositoryArn"]

    def _ensure_execution_role(self, repository_arn: str) -> str:
        scope_source = "\n".join(
            (
                self.config.dataset_s3_uri,
                self.config.checkpoint_s3_uri,
                repository_arn,
            )
        )
        scope = hashlib.sha256(scope_source.encode()).hexdigest()[:8]
        role_name = self.config.execution_role_name or f"{self.config.resource_prefix}-sagemaker-{scope}"
        trust = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        try:
            role = self.iam.get_role(RoleName=role_name)["Role"]
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
                raise
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust),
                Description="MLX SageMaker image-classification training execution role",
                Tags=[{"Key": "mlx:managed", "Value": "true"}],
            )["Role"]
        dataset_bucket, dataset_key = _parse_s3(self.config.dataset_s3_uri)
        checkpoint_bucket, checkpoint_key = _parse_s3(self.config.checkpoint_s3_uri)
        prefix = checkpoint_key.rstrip("/")
        object_arn = f"arn:aws:s3:::{checkpoint_bucket}/{prefix}/*" if prefix else f"arn:aws:s3:::{checkpoint_bucket}/*"
        list_prefix = f"{prefix}/*" if prefix else "*"
        statements: list[dict[str, Any]] = [
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
                "Condition": {"StringLike": {"s3:prefix": [list_prefix]}},
            },
            {
                "Sid": "ManageTrainingArtifacts",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                ],
                "Resource": [object_arn],
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
        ]
        if self.config.kms_key_arn:
            statements.append(
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
            statements.append(
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
            PolicyDocument=json.dumps(
                {"Version": "2012-10-17", "Statement": statements}
            ),
        )
        return role["Arn"]

    def submit(
        self,
        infrastructure: AwsInfrastructure,
        *,
        run_id: Optional[str] = None,
        training: Optional[ImageClassificationRequest] = None,
        training_payload: Optional[Mapping[str, Any]] = None,
        image_uri: Optional[str] = None,
        role_arn: Optional[str] = None,
        resume: bool = False,
    ) -> AwsTrainingSubmission:
        run_id = run_id or uuid4().hex
        request = training or self.config.training
        payload = serialize_training_request(request)
        if training_payload is not None:
            serialized = ImageClassificationRequest.from_config(training_payload)
            if serialized != request:
                raise MLXUserError("The serialized SageMaker training payload does not match the training request.")
            payload = dict(training_payload)
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
        spec = {
            "version": 1,
            "run_id": run_id,
            "region": self.region,
            "run_base_s3_uri": run_base,
            "dataset_s3_uri": self.config.dataset_s3_uri,
            "checkpoint_base_s3_uri": self.config.checkpoint_s3_uri,
            "resource_prefix": self.config.resource_prefix,
            "image_uri": effective_image,
            "role_arn": effective_role,
            "training": payload,
            "immutable_job_config": {
                "ecr_repository": self.config.ecr_repository,
                "execution_role_name": self.config.execution_role_name,
                "network_isolation": self.config.network_isolation,
                "kms_key_arn": self.config.kms_key_arn,
                "vpc": {
                    "subnet_ids": list(self.config.vpc.subnet_ids),
                    "security_group_ids": list(
                        self.config.vpc.security_group_ids
                    ),
                },
                "tags": dict(self.config.tags),
            },
        }
        if not resume:
            self._put_json(run_spec_uri, spec)
        attempt_uri = _join_s3(run_base, "attempts", job_name, "attempt-spec.json")
        self._put_json(attempt_uri, {**spec, "job_name": job_name, "resume": resume})
        stopping: dict[str, Any] = {"MaxRuntimeInSeconds": self.config.max_runtime_seconds}
        if self.config.managed_spot:
            stopping["MaxWaitTimeInSeconds"] = self.config.effective_max_wait_seconds
        create: dict[str, Any] = {
            "TrainingJobName": job_name, "RoleArn": effective_role,
            "AlgorithmSpecification": {
                "TrainingImage": effective_image,
                "TrainingInputMode": "File",
                "EnableSageMakerMetricsTimeSeries": True,
                "MetricDefinitions": [
                    {"Name": "mlx:epoch", "Regex": r"MLX_EPOCH=([0-9.]+);"},
                    {
                        "Name": "mlx:progress",
                        "Regex": r"MLX_PROGRESS=([0-9.]+);",
                    },
                    {
                        "Name": "mlx:eta_seconds",
                        "Regex": r"MLX_ETA_SECONDS=([0-9.]+);",
                    },
                ],
            },
            "InputDataConfig": [
                {
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
                }
            ],
            "OutputDataConfig": {"S3OutputPath": output_uri},
            "ResourceConfig": {
                "InstanceType": self.config.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": self.config.volume_size_gb,
            },
            "StoppingCondition": stopping,
            "EnableManagedSpotTraining": self.config.managed_spot,
            "EnableNetworkIsolation": self.config.network_isolation,
            "CheckpointConfig": {"S3Uri": checkpoint_uri, "LocalPath": "/opt/ml/checkpoints"},
            "HyperParameters": {"mlx_run_id": run_id, "mlx_run_spec_s3_uri": run_spec_uri,
                "mlx_attempt_spec_s3_uri": attempt_uri,
                "mlx_training": json.dumps(payload, separators=(",", ":")),
                "mlx_resume": str(resume).lower(),
                "mlx_image_uri": effective_image,
                "mlx_volume_size_gb": str(self.config.volume_size_gb),
            },
            "Tags": [
                {"Key": "mlx:managed", "Value": "true"},
                {"Key": "mlx:run-id", "Value": run_id},
                *(
                    {"Key": str(key), "Value": str(value)}
                    for key, value in self.config.tags.items()
                ),
            ],
        }
        if self.config.kms_key_arn:
            create["OutputDataConfig"]["KmsKeyId"] = self.config.kms_key_arn
        if self.config.vpc.subnet_ids or self.config.vpc.security_group_ids:
            if not self.config.vpc.subnet_ids or not self.config.vpc.security_group_ids:
                raise MLXUserError(
                    "Both aws.vpc.subnet_ids and security_group_ids are required "
                    "when VPC training is enabled."
                )
            create["VpcConfig"] = {
                "Subnets": list(self.config.vpc.subnet_ids),
                "SecurityGroupIds": list(self.config.vpc.security_group_ids),
            }
        response = self._create_job(create)
        return AwsTrainingSubmission(
            job_name,
            response["TrainingJobArn"],
            run_id,
            "InProgress",
            self.region,
            self.config.managed_spot,
            effective_image,
            checkpoint_uri,
            output_uri,
            self._console_url(job_name),
        )

    def resume(self, job_name: str) -> AwsTrainingSubmission:
        old = self._describe(job_name)
        if old["TrainingJobStatus"] not in {"Stopped", "Failed", "Completed"}:
            raise MLXUserError(f"Training job '{job_name}' must be terminal before it can be resumed.")
        hp = old.get("HyperParameters", {})
        run_id, spec_uri = hp.get("mlx_run_id"), hp.get("mlx_run_spec_s3_uri")
        if not run_id or not spec_uri:
            raise MLXUserError(f"Training job '{job_name}' is not an MLX recoverable job.")
        self._ensure_no_active_attempt(run_id, excluding=job_name)
        spec = self._get_json(spec_uri)
        if spec.get("region") not in {None, self.region}:
            raise MLXUserError(
                "The resume configuration must use the original AWS region."
            )
        if (
            spec.get("dataset_s3_uri") != self.config.dataset_s3_uri
            or spec.get("checkpoint_base_s3_uri")
            != self.config.checkpoint_s3_uri
        ):
            raise MLXUserError(
                "The resume configuration must use the original dataset and "
                "checkpoint S3 locations."
            )
        if spec.get("resource_prefix") != self.config.resource_prefix:
            raise MLXUserError(
                "The resume configuration must use the original aws.resource_prefix."
            )
        if self.config.image_uri and self.config.image_uri != spec.get("image_uri"):
            raise MLXUserError(
                "The resume configuration cannot replace the original immutable image."
            )
        if (
            self.config.execution_role_arn
            and self.config.execution_role_arn != spec.get("role_arn")
        ):
            raise MLXUserError(
                "The resume configuration cannot replace the original execution role."
            )
        expected_job_config = spec.get("immutable_job_config")
        current_job_config = {
            "ecr_repository": self.config.ecr_repository,
            "execution_role_name": self.config.execution_role_name,
            "network_isolation": self.config.network_isolation,
            "kms_key_arn": self.config.kms_key_arn,
            "vpc": {
                "subnet_ids": list(self.config.vpc.subnet_ids),
                "security_group_ids": list(self.config.vpc.security_group_ids),
            },
            "tags": dict(self.config.tags),
        }
        if (
            isinstance(expected_job_config, Mapping)
            and current_job_config != expected_job_config
        ):
            raise MLXUserError(
                "ECR/role names, network isolation, KMS, VPC, and tags cannot "
                "change when resuming a run."
            )
        original_payload = spec.get("training")
        if not isinstance(original_payload, Mapping):
            raise MLXUserError(f"Training job '{job_name}' has an invalid MLX run specification.")
        original = ImageClassificationRequest.from_config(original_payload)
        current = self.config.training
        immutable_old = serialize_training_request(original)
        immutable_new = serialize_training_request(current)
        immutable_old.pop("epochs", None)
        immutable_new.pop("epochs", None)
        normalized_old = json.loads(json.dumps(immutable_old, sort_keys=True))
        normalized_new = json.loads(json.dumps(immutable_new, sort_keys=True))
        if normalized_new != normalized_old:
            raise MLXUserError("Only the total epoch target may change when resuming image-classification training.")
        completed = self._latest_recovery_epoch(spec["run_base_s3_uri"])
        if current.epochs <= original.epochs or current.epochs <= completed:
            raise MLXUserError(
                f"Set a total epoch target above both the original target "
                f"({original.epochs}) and recoverable epoch ({completed})."
            )
        resumed_payload = {**original_payload, "epochs": current.epochs}
        infrastructure = AwsInfrastructure(
            self.region,
            old["TrainingJobArn"].split(":")[4],
            spec["role_arn"],
            spec["image_uri"],
        )
        return self.submit(
            infrastructure,
            run_id=run_id,
            training=ImageClassificationRequest.from_config(resumed_payload),
            training_payload=resumed_payload,
            image_uri=spec["image_uri"],
            role_arn=spec["role_arn"],
            resume=True,
        )

    def _latest_recovery_epoch(self, run_base: str) -> int:
        epochs = []
        for slot in ("a", "b"):
            uri = _join_s3(run_base, "recovery", f"resume-{slot}.json")
            metadata = self._get_optional_json(uri)
            if metadata:
                try:
                    checkpoint_uri = _join_s3(
                        run_base,
                        "recovery",
                        f"resume-{slot}.pth",
                    )
                    bucket, key = _parse_s3(checkpoint_uri)
                    self.s3.head_object(Bucket=bucket, Key=key)
                    epochs.append(int(metadata["epoch"]))
                except self._client_error as exc:
                    if exc.response.get("Error", {}).get("Code") not in {
                        "404",
                        "NoSuchKey",
                        "NotFound",
                    }:
                        raise self._error("recovery checkpoint lookup", exc) from exc
                except self._boto_error as exc:
                    raise self._error("recovery checkpoint lookup", exc) from exc
                except (KeyError, TypeError, ValueError):
                    pass
        if not epochs:
            raise MLXUserError(
                "No recoverable image-classification checkpoint was found under "
                f"{_join_s3(run_base, 'recovery')}."
            )
        return max(epochs)

    def status(self, job_name: str) -> AwsTrainingStatus:
        description = self._describe(job_name)
        return build_training_status(
            description,
            latest_metric=self._latest_metric,
            console_url=self._console_url(job_name),
        )

    def stop(self, job_name: str, *, config_path: str) -> AwsTrainingStopResult:
        current = self.status(job_name)
        status = current.status
        if not current.terminal:
            try:
                self.sagemaker.stop_training_job(TrainingJobName=job_name)
            except (self._client_error, self._boto_error) as exc:
                raise self._error("training job stop", exc) from exc
            status = "Stopping"
        command = (
            "python -m mlx --mode image-classification --platform aws "
            f"--action resume --config {config_path} --job-name {job_name}"
        )
        return AwsTrainingStopResult(job_name, status, current.checkpoint_s3_uri, command)

    def _create_job(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        for index, delay in enumerate((0, 2, 4, 8, 16)):
            if delay:
                time.sleep(delay)
            try:
                return self.sagemaker.create_training_job(**request)
            except self._client_error as exc:
                error = exc.response.get("Error", {})
                message = str(error.get("Message", "")).lower()
                role_propagation = error.get("Code") == "ValidationException" and (
                    "role" in message or "assume" in message
                )
                if not role_propagation or index == 4:
                    raise self._error("training job submission", exc) from exc
            except self._boto_error as exc:
                raise self._error("training job submission", exc) from exc
        raise MLXUserError("AWS training job submission failed after IAM propagation retries.")

    def _ensure_no_active_attempt(self, run_id: str, *, excluding: str) -> None:
        try:
            for state in ("InProgress", "Stopping"):
                paginator = self.sagemaker.get_paginator("list_training_jobs")
                pages = paginator.paginate(
                    StatusEquals=state,
                    NameContains=_safe_name(self.config.resource_prefix)[:32],
                )
                for page in pages:
                    for summary in page.get("TrainingJobSummaries", []):
                        candidate = summary["TrainingJobName"]
                        candidate_run = self._describe(candidate).get(
                            "HyperParameters", {}
                        ).get("mlx_run_id")
                        if candidate != excluding and candidate_run == run_id:
                            raise MLXUserError(
                                f"Logical run '{run_id}' already has active SageMaker "
                                f"job '{candidate}'."
                            )
        except (self._client_error, self._boto_error) as exc:
            raise self._error("active training attempt lookup", exc) from exc

    def _new_job_name(self, run_id: str) -> str:
        prefix = _safe_name(self.config.resource_prefix)[:32]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{timestamp}-{run_id[:8]}-{uuid4().hex[:4]}"[:63].rstrip(
            "-"
        )

    def _put_json(self, uri: str, value: Mapping[str, Any]) -> None:
        bucket, key = _parse_s3(uri)
        try:
            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(value, sort_keys=True).encode(),
                ContentType="application/json",
            )
        except (self._client_error, self._boto_error) as exc:
            raise self._error("run manifest upload", exc) from exc

    def _get_json(self, uri: str) -> Mapping[str, Any]:
        bucket, key = _parse_s3(uri)
        try:
            body = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            value = json.loads(body)
        except (self._client_error, self._boto_error, json.JSONDecodeError) as exc:
            raise self._error("run manifest retrieval", exc) from exc
        if not isinstance(value, Mapping):
            raise MLXUserError(f"AWS run manifest is invalid: {uri}")
        return value

    def _get_optional_json(self, uri: str) -> Optional[Mapping[str, Any]]:
        bucket, key = _parse_s3(uri)
        try:
            body = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return None
            raise self._error("recovery metadata retrieval", exc) from exc
        except self._boto_error as exc:
            raise self._error("recovery metadata retrieval", exc) from exc
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, Mapping) else None

    def _describe(self, job_name: str) -> Mapping[str, Any]:
        try:
            return self.sagemaker.describe_training_job(TrainingJobName=job_name)
        except (self._client_error, self._boto_error) as exc:
            raise self._error(f"training job lookup for '{job_name}'", exc) from exc

    def _latest_metric(self, description: Mapping[str, Any], metric: str) -> Optional[float]:
        creation = description.get("CreationTime")
        if creation is None:
            return None
        try:
            response = self.cloudwatch.get_metric_statistics(Namespace="/aws/sagemaker/TrainingJobs", MetricName=metric,
                Dimensions=[{"Name": "TrainingJobName", "Value": description["TrainingJobName"]}], StartTime=creation,
                EndTime=datetime.now(timezone.utc), Period=60, Statistics=["Maximum"])
        except (self._client_error, self._boto_error):
            return None
        points = response.get("Datapoints", [])
        return float(max(points, key=lambda item: item["Timestamp"])["Maximum"]) if points else None

    def _console_url(self, job_name: str) -> str:
        return (
            f"https://{self.region}.console.aws.amazon.com/sagemaker/home"
            f"?region={self.region}#/jobs/{quote(job_name)}"
        )
