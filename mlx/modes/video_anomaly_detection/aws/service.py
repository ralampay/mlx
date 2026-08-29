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

from mlx.core.aws.clients import AwsClientBundle
from mlx.core.aws.image import PublishSageMakerImage
from mlx.core.aws.models import AwsInfrastructure
from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.aws.config import AwsVideoAnomalyTrainingConfig
from mlx.modes.video_anomaly_detection.aws.models import (
    AwsVideoAnomalyBatchStatus,
    AwsVideoAnomalyBatchSubmission,
    VideoAnomalyVariantStatus,
)
from mlx.modes.video_anomaly_detection.variants import video_anomaly_model_variants


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _join_s3(uri: str, *parts: str) -> str:
    return "/".join([uri.rstrip("/"), *(part.strip("/") for part in parts)])


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-") or "mlx-vad"


class SageMakerVideoAnomalyBatchService:
    def __init__(
        self,
        config: AwsVideoAnomalyTrainingConfig,
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
        self._client_error = bundle.client_error
        self._boto_error = bundle.boto_error

    def _error(self, action: str, exc: Exception) -> MLXUserError:
        return MLXUserError(f"AWS video-anomaly {action} failed: {exc}")

    def validate_storage(self) -> None:
        dataset_bucket, dataset_key = _parse_s3(self.config.dataset_s3_uri)
        output_bucket, _ = _parse_s3(self.config.output_s3_uri)
        try:
            self.s3.head_object(Bucket=dataset_bucket, Key=dataset_key)
            self.s3.head_bucket(Bucket=output_bucket)
            locations = (
                ("dataset", self._bucket_region(dataset_bucket)),
                ("output", self._bucket_region(output_bucket)),
            )
        except (self._client_error, self._boto_error) as exc:
            raise self._error("S3 validation", exc) from exc
        for label, location in locations:
            if location != self.region:
                raise MLXUserError(
                    f"The {label} S3 location is in region '{location}', but the "
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
                name, uri, repository_arn = self._ensure_repository()
                image_uri = PublishSageMakerImage(
                    ecr=self.ecr,
                    repository_name=name,
                    repository_uri=uri,
                    package_root=Path(__file__).resolve().parents[3],
                    dockerfile=Path(__file__).with_name("Dockerfile"),
                    rebuild=self.config.rebuild_image,
                    client_error=self._client_error,
                    boto_error=self._boto_error,
                ).execute()
            role = self.config.execution_role_arn or self._ensure_execution_role(repository_arn)
        except MLXUserError:
            raise
        except (self._client_error, self._boto_error) as exc:
            raise self._error("infrastructure preparation", exc) from exc
        return AwsInfrastructure(self.region, account, role, image_uri)

    def _repository_arn(self, image_uri: str) -> str:
        match = re.match(
            r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
            r"(?P<repository>[^:@]+)(?::[^@]+|@sha256:[a-f0-9]+)?$",
            image_uri,
        )
        if not match:
            raise MLXUserError(
                "aws.image_uri must be an Amazon ECR image when MLX creates the execution role; "
                "otherwise provide aws.execution_role_arn as well."
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
                repositoryName=name,
                imageTagMutability="IMMUTABLE",
                imageScanningConfiguration={"scanOnPush": True},
                tags=[{"Key": "mlx:managed", "Value": "true"}],
            )["repository"]
        return name, repository["repositoryUri"], repository["repositoryArn"]

    def _ensure_execution_role(self, repository_arn: str) -> str:
        scope = hashlib.sha256(
            "\n".join((self.config.dataset_s3_uri, self.config.output_s3_uri, repository_arn)).encode()
        ).hexdigest()[:8]
        role_name = self.config.execution_role_name or f"{self.config.resource_prefix}-sagemaker-{scope}"
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
                Description="MLX SageMaker video-anomaly batch training execution role",
                Tags=[{"Key": "mlx:managed", "Value": "true"}],
            )["Role"]
        dataset_bucket, dataset_key = _parse_s3(self.config.dataset_s3_uri)
        output_bucket, output_key = _parse_s3(self.config.output_s3_uri)
        output_prefix = output_key.rstrip("/")
        output_object_arn = (
            f"arn:aws:s3:::{output_bucket}/{output_prefix}/*"
            if output_prefix
            else f"arn:aws:s3:::{output_bucket}/*"
        )
        statements: list[dict[str, Any]] = [
            {
                "Sid": "ReadDataset", "Effect": "Allow", "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{dataset_bucket}/{dataset_key}"],
            },
            {
                "Sid": "InspectBuckets", "Effect": "Allow", "Action": ["s3:GetBucketLocation"],
                "Resource": [f"arn:aws:s3:::{dataset_bucket}", f"arn:aws:s3:::{output_bucket}"],
            },
            {
                "Sid": "ListTrainingData", "Effect": "Allow", "Action": ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{dataset_bucket}", f"arn:aws:s3:::{output_bucket}"],
            },
            {
                "Sid": "ManageArtifacts", "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
                "Resource": [output_object_arn],
            },
            {
                "Sid": "PullTrainingImage", "Effect": "Allow",
                "Action": ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                "Resource": [repository_arn],
            },
            {"Sid": "EcrAuthorization", "Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": ["*"]},
            {
                "Sid": "TrainingLogs", "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:DescribeLogStreams", "logs:PutLogEvents", "cloudwatch:PutMetricData"],
                "Resource": ["*"],
            },
        ]
        if self.config.kms_key_arn:
            statements.append({
                "Sid": "TrainingKmsKey", "Effect": "Allow",
                "Action": ["kms:Decrypt", "kms:DescribeKey", "kms:Encrypt", "kms:GenerateDataKey"],
                "Resource": [self.config.kms_key_arn],
            })
        if self.config.vpc.subnet_ids or self.config.vpc.security_group_ids:
            statements.append({
                "Sid": "TrainingVpcNetworkInterfaces", "Effect": "Allow",
                "Action": ["ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface", "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeDhcpOptions", "ec2:DescribeNetworkInterfaces", "ec2:DescribeSecurityGroups", "ec2:DescribeSubnets", "ec2:DescribeVpcs"],
                "Resource": ["*"],
            })
        self.iam.put_role_policy(
            RoleName=role_name,
            PolicyName="MLXSageMakerTraining",
            PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}),
        )
        return role["Arn"]

    def submit(
        self,
        infrastructure: AwsInfrastructure,
        *,
        batch_id: Optional[str] = None,
        image_uri: Optional[str] = None,
        role_arn: Optional[str] = None,
        resume: bool = False,
        frozen_spec: Optional[Mapping[str, Any]] = None,
    ) -> AwsVideoAnomalyBatchSubmission:
        batch_id = batch_id or uuid4().hex
        variants = [item.to_dict() for item in video_anomaly_model_variants()]
        effective_image = image_uri or infrastructure.image_uri
        effective_role = role_arn or infrastructure.role_arn
        batch_uri = _join_s3(
            self.config.output_s3_uri, self.config.resource_prefix, "batches", batch_id
        )
        spec_uri = _join_s3(batch_uri, "batch-spec.json")
        spec = dict(frozen_spec or {
            "version": 1,
            "batch_id": batch_id,
            "region": self.region,
            "batch_s3_uri": batch_uri,
            "dataset_s3_uri": self.config.dataset_s3_uri,
            "output_base_s3_uri": self.config.output_s3_uri,
            "resource_prefix": self.config.resource_prefix,
            "image_uri": effective_image,
            "role_arn": effective_role,
            "training": dict(self.config.training),
            "variants": variants,
            "immutable_job_config": self._immutable_job_config(),
        })
        if not resume:
            self._put_json(spec_uri, spec)
        job_name = self._new_job_name(batch_id)
        attempt_uri = _join_s3(
            self.config.output_s3_uri,
            self.config.resource_prefix,
            "attempts",
            batch_id,
            job_name,
        )
        self._put_json(
            _join_s3(attempt_uri, "attempt-spec.json"),
            {**spec, "job_name": job_name, "resume": resume},
        )
        stopping: dict[str, Any] = {"MaxRuntimeInSeconds": self.config.max_runtime_seconds}
        if self.config.managed_spot:
            stopping["MaxWaitTimeInSeconds"] = self.config.effective_max_wait_seconds
        request: dict[str, Any] = {
            "TrainingJobName": job_name,
            "RoleArn": effective_role,
            "AlgorithmSpecification": {
                "TrainingImage": effective_image,
                "TrainingInputMode": "File",
                "EnableSageMakerMetricsTimeSeries": True,
                "MetricDefinitions": [
                    {"Name": "mlx:model_index", "Regex": r"MLX_MODEL_INDEX=([0-9.]+);"},
                    {"Name": "mlx:epoch", "Regex": r"MLX_EPOCH=([0-9.]+);"},
                    {"Name": "mlx:progress", "Regex": r"MLX_PROGRESS=([0-9.]+);"},
                ],
            },
            "InputDataConfig": [{
                "ChannelName": "training",
                "ContentType": "application/zip",
                "InputMode": "File",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": self.config.dataset_s3_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }},
            }],
            "OutputDataConfig": {"S3OutputPath": attempt_uri},
            "ResourceConfig": {
                "InstanceType": self.config.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": self.config.volume_size_gb,
            },
            "StoppingCondition": stopping,
            "EnableManagedSpotTraining": self.config.managed_spot,
            "EnableNetworkIsolation": self.config.network_isolation,
            "CheckpointConfig": {"S3Uri": batch_uri, "LocalPath": "/opt/ml/checkpoints"},
            "HyperParameters": {
                "mlx_batch_id": batch_id,
                "mlx_batch_spec_s3_uri": spec_uri,
                "mlx_training": json.dumps(spec["training"], separators=(",", ":")),
                "mlx_variants": json.dumps(spec["variants"], separators=(",", ":")),
                "mlx_resume": str(resume).lower(),
                "mlx_image_uri": effective_image,
                "mlx_volume_size_gb": str(self.config.volume_size_gb),
            },
            "Tags": [
                {"Key": "mlx:managed", "Value": "true"},
                {"Key": "mlx:batch-id", "Value": batch_id},
                *({"Key": str(key), "Value": str(value)} for key, value in self.config.tags.items()),
            ],
        }
        if self.config.kms_key_arn:
            request["OutputDataConfig"]["KmsKeyId"] = self.config.kms_key_arn
        if self.config.vpc.subnet_ids or self.config.vpc.security_group_ids:
            if not self.config.vpc.subnet_ids or not self.config.vpc.security_group_ids:
                raise MLXUserError("Both subnet_ids and security_group_ids are required for VPC training.")
            request["VpcConfig"] = {
                "Subnets": list(self.config.vpc.subnet_ids),
                "SecurityGroupIds": list(self.config.vpc.security_group_ids),
            }
        response = self._create_job(request)
        return AwsVideoAnomalyBatchSubmission(
            job_name=job_name,
            job_arn=response["TrainingJobArn"],
            batch_id=batch_id,
            status="InProgress",
            region=self.region,
            managed_spot=self.config.managed_spot,
            image_uri=effective_image,
            model_count=len(spec["variants"]),
            batch_s3_uri=batch_uri,
            output_s3_uri=attempt_uri,
            console_url=self._console_url(job_name),
        )

    def resume(self, job_name: str) -> AwsVideoAnomalyBatchSubmission:
        old = self._describe(job_name)
        if old["TrainingJobStatus"] not in {"Stopped", "Failed", "Completed"}:
            raise MLXUserError(f"Training job '{job_name}' must be terminal before it can be resumed.")
        hyperparameters = old.get("HyperParameters", {})
        batch_id = hyperparameters.get("mlx_batch_id")
        spec_uri = hyperparameters.get("mlx_batch_spec_s3_uri")
        if not batch_id or not spec_uri:
            raise MLXUserError(f"Training job '{job_name}' is not an MLX video-anomaly batch.")
        self._ensure_no_active_attempt(str(batch_id), excluding=job_name)
        spec = self._get_json(str(spec_uri))
        expected = {
            "region": self.region,
            "dataset_s3_uri": self.config.dataset_s3_uri,
            "output_base_s3_uri": self.config.output_s3_uri,
            "resource_prefix": self.config.resource_prefix,
            "training": dict(self.config.training),
            "immutable_job_config": self._immutable_job_config(),
        }
        mismatches = [key for key, value in expected.items() if spec.get(key) != value]
        if mismatches:
            raise MLXUserError(
                "Resume configuration differs from the frozen batch spec: "
                + ", ".join(mismatches)
                + "."
            )
        infrastructure = AwsInfrastructure(
            self.region,
            old["TrainingJobArn"].split(":")[4],
            str(spec["role_arn"]),
            str(spec["image_uri"]),
        )
        return self.submit(
            infrastructure,
            batch_id=str(batch_id),
            image_uri=str(spec["image_uri"]),
            role_arn=str(spec["role_arn"]),
            resume=True,
            frozen_spec=spec,
        )

    def status(self, job_name: str) -> AwsVideoAnomalyBatchStatus:
        description = self._describe(job_name)
        hyperparameters = description.get("HyperParameters", {})
        batch_id = hyperparameters.get("mlx_batch_id")
        checkpoint_uri = description.get("CheckpointConfig", {}).get("S3Uri")
        manifest = self._get_optional_json(_join_s3(checkpoint_uri, "batch-status.json")) if checkpoint_uri else None
        spec_uri = hyperparameters.get("mlx_batch_spec_s3_uri")
        spec = self._get_optional_json(str(spec_uri)) if spec_uri and manifest is None else None
        variant_values = manifest.get("variants", ()) if manifest else spec.get("variants", ()) if spec else ()
        fallback_epochs = int(spec.get("training", {}).get("epochs", 0)) if spec else 0
        variants = tuple(
            VideoAnomalyVariantStatus(
                variant_id=str(item.get("variant_id")),
                model_name=str(item.get("model_name")),
                drax_fusion_mode=item.get("drax_fusion_mode"),
                status=str(item.get("status", "pending")),
                completed_epoch=int(item.get("completed_epoch", 0)),
                total_epochs=int(item.get("total_epochs", fallback_epochs)),
                error=str(item["error"]) if item.get("error") else None,
            )
            for item in variant_values
        )
        completed = sum(item.status == "completed" for item in variants)
        fractional = sum(
            item.completed_epoch / item.total_epochs
            for item in variants
            if item.status == "running" and item.total_epochs
        )
        total = len(variants)
        return AwsVideoAnomalyBatchStatus(
            job_name=job_name,
            batch_id=str(batch_id) if batch_id else None,
            status=str(description["TrainingJobStatus"]),
            current_variant=str(manifest.get("current_variant")) if manifest and manifest.get("current_variant") else None,
            completed_models=completed,
            total_models=total,
            progress_percent=(min(100.0, (completed + fractional) / total * 100) if total else None),
            variants=variants,
            batch_s3_uri=checkpoint_uri,
            output_s3_uri=(description.get("ModelArtifacts", {}).get("S3ModelArtifacts") or description.get("OutputDataConfig", {}).get("S3OutputPath")),
            failure_reason=(description.get("FailureReason") or (manifest.get("failure_reason") if manifest else None)),
            console_url=self._console_url(job_name),
        )

    def _immutable_job_config(self) -> dict[str, Any]:
        return {
            "ecr_repository": self.config.ecr_repository,
            "instance_type": self.config.instance_type,
            "volume_size_gb": self.config.volume_size_gb,
            "managed_spot": self.config.managed_spot,
            "max_runtime_seconds": self.config.max_runtime_seconds,
            "max_wait_seconds": self.config.max_wait_seconds,
            "execution_role_name": self.config.execution_role_name,
            "execution_role_arn": self.config.execution_role_arn,
            "image_uri": self.config.image_uri,
            "network_isolation": self.config.network_isolation,
            "kms_key_arn": self.config.kms_key_arn,
            "vpc": {
                "subnet_ids": list(self.config.vpc.subnet_ids),
                "security_group_ids": list(self.config.vpc.security_group_ids),
            },
            "tags": dict(self.config.tags),
        }

    def _create_job(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        for index, delay in enumerate((0, 2, 4, 8, 16)):
            if delay:
                time.sleep(delay)
            try:
                return self.sagemaker.create_training_job(**request)
            except self._client_error as exc:
                message = str(exc.response.get("Error", {}).get("Message", "")).lower()
                retry = exc.response.get("Error", {}).get("Code") == "ValidationException" and ("role" in message or "assume" in message)
                if not retry or index == 4:
                    raise self._error("training job submission", exc) from exc
            except self._boto_error as exc:
                raise self._error("training job submission", exc) from exc
        raise MLXUserError("AWS training job submission failed after IAM propagation retries.")

    def _ensure_no_active_attempt(self, batch_id: str, *, excluding: str) -> None:
        try:
            paginator = self.sagemaker.get_paginator("list_training_jobs")
            for state in ("InProgress", "Stopping"):
                for page in paginator.paginate(StatusEquals=state, NameContains=_safe_name(self.config.resource_prefix)[:32]):
                    for summary in page.get("TrainingJobSummaries", []):
                        candidate = summary["TrainingJobName"]
                        candidate_id = self._describe(candidate).get("HyperParameters", {}).get("mlx_batch_id")
                        if candidate != excluding and candidate_id == batch_id:
                            raise MLXUserError(f"Batch '{batch_id}' already has active job '{candidate}'.")
        except (self._client_error, self._boto_error) as exc:
            raise self._error("active training attempt lookup", exc) from exc

    def _new_job_name(self, batch_id: str) -> str:
        prefix = _safe_name(self.config.resource_prefix)[:32]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{timestamp}-{batch_id[:8]}-{uuid4().hex[:4]}"[:63].rstrip("-")

    def _put_json(self, uri: str, value: Mapping[str, Any]) -> None:
        bucket, key = _parse_s3(uri)
        try:
            self.s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(value, sort_keys=True).encode(), ContentType="application/json")
        except (self._client_error, self._boto_error) as exc:
            raise self._error("manifest upload", exc) from exc

    def _get_json(self, uri: str) -> Mapping[str, Any]:
        bucket, key = _parse_s3(uri)
        try:
            value = json.loads(self.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        except (self._client_error, self._boto_error, json.JSONDecodeError) as exc:
            raise self._error("manifest retrieval", exc) from exc
        if not isinstance(value, Mapping):
            raise MLXUserError(f"AWS batch manifest is invalid: {uri}")
        return value

    def _get_optional_json(self, uri: str) -> Optional[Mapping[str, Any]]:
        bucket, key = _parse_s3(uri)
        try:
            body = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise self._error("batch status retrieval", exc) from exc
        except self._boto_error as exc:
            raise self._error("batch status retrieval", exc) from exc
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

    def _console_url(self, job_name: str) -> str:
        return f"https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/jobs/{quote(job_name)}"


__all__ = ["SageMakerVideoAnomalyBatchService"]
