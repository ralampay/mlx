from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest


_AWS_KEYS = {
    "region",
    "profile",
    "dataset_s3_uri",
    "checkpoint_s3_uri",
    "instance_type",
    "volume_size_gb",
    "managed_spot",
    "max_runtime_seconds",
    "max_wait_seconds",
    "resource_prefix",
    "ecr_repository",
    "execution_role_name",
    "execution_role_arn",
    "image_uri",
    "network_isolation",
    "kms_key_arn",
    "vpc",
    "tags",
}
_VPC_KEYS = {"subnet_ids", "security_group_ids"}
_TRAINING_KEYS = set(TrainObjectDetectionRequest.__dataclass_fields__) - {
    "dataset_path",
    "dataset_s3_uri",
    "dataset_cache_dir",
    "output_path",
    "model_path",
}
_CLI_TRAINING_KEYS = _TRAINING_KEYS - {"plots", "save_period"}


@dataclass(frozen=True)
class AwsVpcConfig:
    subnet_ids: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AwsTrainingConfig:
    dataset_s3_uri: str
    checkpoint_s3_uri: str
    instance_type: str
    training: TrainObjectDetectionRequest
    region: Optional[str] = None
    profile: Optional[str] = None
    volume_size_gb: int = 100
    managed_spot: bool = True
    max_runtime_seconds: int = 86400
    max_wait_seconds: Optional[int] = None
    resource_prefix: str = "mlx-od"
    ecr_repository: Optional[str] = None
    execution_role_name: Optional[str] = None
    execution_role_arn: Optional[str] = None
    image_uri: Optional[str] = None
    network_isolation: bool = False
    kms_key_arn: Optional[str] = None
    vpc: AwsVpcConfig = field(default_factory=AwsVpcConfig)
    tags: Mapping[str, str] = field(default_factory=dict)
    rebuild_image: bool = False

    @property
    def effective_max_wait_seconds(self) -> Optional[int]:
        if not self.managed_spot:
            return None
        if self.max_wait_seconds is not None:
            return self.max_wait_seconds
        return min(self.max_runtime_seconds * 2, 30 * 24 * 60 * 60)

    def to_run_spec(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset_s3_uri": self.dataset_s3_uri,
            "checkpoint_s3_uri": self.checkpoint_s3_uri,
            "training": self.training.to_config(),
        }


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise MLXUserError(
            "AWS configuration requires PyYAML. Install MLX with the 'aws' extra."
        ) from exc

    if not path.exists():
        raise MLXUserError(f"AWS configuration file not found: {path}")
    if not path.is_file():
        raise MLXUserError(f"AWS configuration path is not a file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MLXUserError(f"Invalid AWS YAML configuration: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MLXUserError("AWS configuration must contain a YAML mapping at its root.")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise MLXUserError(f"Unknown {label} configuration key(s): {', '.join(unknown)}.")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MLXUserError(f"'{label}' must be a YAML mapping.")
    return value


def _validate_s3_uri(
    value: str,
    label: str,
    *,
    zip_object: bool = False,
    allow_bucket_root: bool = False,
) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or (not allow_bucket_root and not parsed.path.strip("/"))
    ):
        expected = "s3://bucket or s3://bucket/prefix" if allow_bucket_root else "s3://bucket/key"
        raise MLXUserError(f"'{label}' must be a {expected} URI.")
    normalized = value.rstrip("/")
    if zip_object and not normalized.lower().endswith(".zip"):
        raise MLXUserError(f"'{label}' must point to a .zip dataset object.")
    return normalized


def load_aws_training_config(
    config_path: str,
    cli_config: Mapping[str, Any],
) -> AwsTrainingConfig:
    root = _load_yaml(Path(config_path).expanduser())
    _reject_unknown(root, {"version", "aws", "training"}, "root")
    if root.get("version", 1) != 1:
        raise MLXUserError("Unsupported AWS configuration version; expected version: 1.")

    aws = _require_mapping(root.get("aws"), "aws")
    training = dict(_require_mapping(root.get("training"), "training"))
    _reject_unknown(aws, _AWS_KEYS, "aws")
    _reject_unknown(training, _TRAINING_KEYS, "training")

    vpc_values = _require_mapping(aws.get("vpc"), "aws.vpc")
    _reject_unknown(vpc_values, _VPC_KEYS, "aws.vpc")
    tags = _require_mapping(aws.get("tags"), "aws.tags")

    explicit = set(cli_config.get("_explicit_options") or ())
    for key in _CLI_TRAINING_KEYS:
        if key in explicit and key in cli_config:
            training[key] = cli_config[key]
    if "instance_type" in explicit:
        aws = {**aws, "instance_type": cli_config.get("instance_type")}
    if "profile" in explicit:
        aws = {**aws, "profile": cli_config.get("profile")}
    if "dataset_s3_uri" in explicit:
        action = str(cli_config.get("action") or "train")
        if action not in {"train", "resume"}:
            raise MLXUserError(
                "--dataset-s3-uri is supported only for AWS train or resume actions."
            )
        if "dataset_path" in explicit:
            raise MLXUserError("Use either --dataset or --dataset-s3-uri, not both.")
        aws = {**aws, "dataset_s3_uri": cli_config.get("dataset_s3_uri")}

    training.setdefault("provider", "ultralytics")
    training.setdefault("device", "auto")
    training.setdefault("save_period", -1)
    if not training.get("model"):
        raise MLXUserError("AWS object-detection training requires 'training.model'.")

    dataset_uri = _validate_s3_uri(
        str(aws.get("dataset_s3_uri") or ""),
        "aws.dataset_s3_uri",
        zip_object=True,
    )
    checkpoint_uri = _validate_s3_uri(
        str(aws.get("checkpoint_s3_uri") or ""),
        "aws.checkpoint_s3_uri",
        allow_bucket_root=True,
    )
    instance_type = str(aws.get("instance_type") or "").strip()
    if not instance_type:
        raise MLXUserError("AWS configuration requires 'aws.instance_type'.")
    volume_size_gb = int(aws.get("volume_size_gb", 100))
    if volume_size_gb < 1:
        raise MLXUserError("aws.volume_size_gb must be at least 1.")

    max_runtime = int(aws.get("max_runtime_seconds", 86400))
    if max_runtime <= 0 or max_runtime > 28 * 24 * 60 * 60:
        raise MLXUserError("aws.max_runtime_seconds must be between 1 second and 28 days.")
    managed_spot = bool(aws.get("managed_spot", True))
    raw_max_wait = aws.get("max_wait_seconds")
    max_wait = int(raw_max_wait) if raw_max_wait is not None else None
    effective_wait = max_wait or min(max_runtime * 2, 30 * 24 * 60 * 60)
    if managed_spot and effective_wait < max_runtime:
        raise MLXUserError(
            "aws.max_wait_seconds must be greater than or equal to max_runtime_seconds."
        )

    resource_prefix = str(aws.get("resource_prefix") or "mlx-od").strip().lower()
    if not resource_prefix or len(resource_prefix) > 40:
        raise MLXUserError("aws.resource_prefix must contain 1-40 characters.")

    request = TrainObjectDetectionRequest.from_config(training)
    return AwsTrainingConfig(
        dataset_s3_uri=dataset_uri,
        checkpoint_s3_uri=checkpoint_uri,
        instance_type=instance_type,
        training=request,
        region=str(aws["region"]) if aws.get("region") else None,
        profile=str(aws["profile"]) if aws.get("profile") else None,
        volume_size_gb=volume_size_gb,
        managed_spot=managed_spot,
        max_runtime_seconds=max_runtime,
        max_wait_seconds=max_wait,
        resource_prefix=resource_prefix,
        ecr_repository=str(aws["ecr_repository"]) if aws.get("ecr_repository") else None,
        execution_role_name=(
            str(aws["execution_role_name"]) if aws.get("execution_role_name") else None
        ),
        execution_role_arn=(
            str(aws["execution_role_arn"]) if aws.get("execution_role_arn") else None
        ),
        image_uri=str(aws["image_uri"]) if aws.get("image_uri") else None,
        network_isolation=bool(aws.get("network_isolation", False)),
        kms_key_arn=str(aws["kms_key_arn"]) if aws.get("kms_key_arn") else None,
        vpc=AwsVpcConfig(
            subnet_ids=tuple(str(item) for item in vpc_values.get("subnet_ids", ())),
            security_group_ids=tuple(
                str(item) for item in vpc_values.get("security_group_ids", ())
            ),
        ),
        tags={str(key): str(value) for key, value in tags.items()},
        rebuild_image=bool(cli_config.get("rebuild_image", False)),
    )
