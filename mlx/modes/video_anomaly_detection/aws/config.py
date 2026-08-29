from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.requests import TrainVideoAnomalyRequest


_AWS_KEYS = {
    "region", "profile", "dataset_s3_uri", "output_s3_uri", "instance_type",
    "volume_size_gb", "managed_spot", "max_runtime_seconds", "max_wait_seconds",
    "resource_prefix", "ecr_repository", "execution_role_name", "execution_role_arn",
    "image_uri", "network_isolation", "kms_key_arn", "vpc", "tags",
}
_VPC_KEYS = {"subnet_ids", "security_group_ids"}
_EXCLUDED_TRAINING_KEYS = {
    "model", "model_path", "dataset_path", "dataset_s3_uri", "dataset_cache_dir",
    "output_path", "drax_fusion_mode", "backbone_mode_explicit",
    "temporal_options_explicit", "backbone_mode", "temporal_model",
    "temporal_hidden_dim", "temporal_embedding_dim", "temporal_kernel_size",
    "temporal_dropout", "extras",
}
_TRAINING_KEYS = set(TrainVideoAnomalyRequest.__dataclass_fields__) - _EXCLUDED_TRAINING_KEYS
_CLI_ARCHITECTURE_KEYS = {
    "model", "drax_fusion_mode", "backbone_mode", "temporal_model",
    "temporal_hidden_dim", "temporal_embedding_dim", "temporal_kernel_size",
    "temporal_dropout",
}


@dataclass(frozen=True)
class AwsVpcConfig:
    subnet_ids: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AwsVideoAnomalyTrainingConfig:
    dataset_s3_uri: str
    output_s3_uri: str
    instance_type: str
    training: Mapping[str, Any]
    region: Optional[str] = None
    profile: Optional[str] = None
    volume_size_gb: int = 200
    managed_spot: bool = True
    max_runtime_seconds: int = 604800
    max_wait_seconds: Optional[int] = None
    resource_prefix: str = "mlx-vad"
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
        return self.max_wait_seconds or min(
            self.max_runtime_seconds * 2, 30 * 24 * 60 * 60
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MLXUserError(f"'{label}' must be a YAML mapping.")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MLXUserError(f"Unknown {label} configuration key(s): {', '.join(unknown)}.")


def _s3_uri(
    value: Any,
    label: str,
    *,
    zip_object: bool = False,
    allow_bucket_root: bool = False,
) -> str:
    normalized = str(value or "").rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or (not allow_bucket_root and not parsed.path.strip("/"))
    ):
        expected = "s3://bucket or s3://bucket/prefix" if allow_bucket_root else "s3://bucket/key"
        raise MLXUserError(f"'{label}' must be a valid {expected} URI.")
    if zip_object and not normalized.lower().endswith(".zip"):
        raise MLXUserError(f"'{label}' must point to a .zip dataset object.")
    return normalized


def serialize_training_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(_TRAINING_KEYS) if key in value}


def load_aws_training_config(
    config_path: str,
    cli_config: Mapping[str, Any],
) -> AwsVideoAnomalyTrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise MLXUserError(
            "AWS configuration requires PyYAML. Install MLX with the 'aws' extra."
        ) from exc
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise MLXUserError(f"AWS configuration file not found: {path}")
    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MLXUserError(f"Invalid AWS YAML configuration: {exc}") from exc
    root = _mapping(root, "root")
    _reject_unknown(root, {"version", "aws", "training"}, "root")
    if root.get("version", 1) != 1:
        raise MLXUserError("Unsupported AWS configuration version; expected version: 1.")
    aws = dict(_mapping(root.get("aws"), "aws"))
    training = dict(_mapping(root.get("training"), "training"))
    _reject_unknown(aws, _AWS_KEYS, "aws")
    _reject_unknown(training, _TRAINING_KEYS, "training")
    vpc = _mapping(aws.get("vpc"), "aws.vpc")
    tags = _mapping(aws.get("tags"), "aws.tags")
    _reject_unknown(vpc, _VPC_KEYS, "aws.vpc")

    explicit = set(cli_config.get("_explicit_options") or ())
    explicit_architecture = sorted(explicit & _CLI_ARCHITECTURE_KEYS)
    if explicit_architecture:
        raise MLXUserError(
            "AWS video-anomaly train-all selects the complete 3D model inventory; "
            "do not pass architecture option(s): "
            + ", ".join(explicit_architecture)
            + "."
        )
    for key in _TRAINING_KEYS & explicit:
        if key in cli_config:
            training[key] = cli_config[key]
    for key in ("instance_type", "profile", "dataset_s3_uri"):
        if key in explicit:
            aws[key] = cli_config.get(key)

    if training.get("device") not in {None, "auto"}:
        raise MLXUserError("AWS video-anomaly train-all requires training.device: auto.")
    defaults = TrainVideoAnomalyRequest(model="resnet18").to_config()
    training = {**serialize_training_config(defaults), **training}
    training.update({"device": "auto", "backbone_mode": "3d"})
    probe = TrainVideoAnomalyRequest.from_config({**training, "model": "resnet18"})
    if probe.epochs < 1 or probe.batch_size < 1:
        raise MLXUserError("training.epochs and training.batch_size must be at least 1.")
    if probe.height < 1 or probe.width < 1 or probe.workers < 0:
        raise MLXUserError(
            "training.height/width must be positive and training.workers cannot be negative."
        )
    if probe.clip_length < 1 or probe.frame_stride < 1:
        raise MLXUserError("training.clip_length and training.frame_stride must be at least 1.")
    if probe.svdd_dim < 1 or probe.svdd_hidden_dim < 1:
        raise MLXUserError("training.svdd_dim and training.svdd_hidden_dim must be at least 1.")
    if probe.lr is None or probe.lr <= 0:
        raise MLXUserError("training.lr must be greater than zero.")
    if probe.backbone_temporal_kernel_size < 1 or probe.backbone_temporal_kernel_size % 2 == 0:
        raise MLXUserError("training.backbone_temporal_kernel_size must be positive and odd.")
    if probe.clip_length < probe.backbone_temporal_kernel_size:
        raise MLXUserError("training.clip_length must be at least the temporal kernel size.")
    if not 0 < probe.svdd_quantile < 1:
        raise MLXUserError("training.svdd_quantile must be strictly between zero and one.")

    instance_type = str(aws.get("instance_type") or "").strip()
    if not instance_type:
        raise MLXUserError("AWS configuration requires 'aws.instance_type'.")
    volume = int(aws.get("volume_size_gb", 200))
    runtime = int(aws.get("max_runtime_seconds", 604800))
    if volume < 1:
        raise MLXUserError("aws.volume_size_gb must be at least 1.")
    if runtime < 1 or runtime > 28 * 24 * 60 * 60:
        raise MLXUserError("aws.max_runtime_seconds must be between 1 second and 28 days.")
    managed_spot = bool(aws.get("managed_spot", True))
    wait = int(aws["max_wait_seconds"]) if aws.get("max_wait_seconds") is not None else None
    effective_wait = wait or min(runtime * 2, 30 * 24 * 60 * 60)
    if managed_spot and effective_wait < runtime:
        raise MLXUserError("aws.max_wait_seconds must be at least max_runtime_seconds.")
    prefix = str(aws.get("resource_prefix") or "mlx-vad").strip().lower()
    if not prefix or len(prefix) > 40:
        raise MLXUserError("aws.resource_prefix must contain 1-40 characters.")

    return AwsVideoAnomalyTrainingConfig(
        dataset_s3_uri=_s3_uri(aws.get("dataset_s3_uri"), "aws.dataset_s3_uri", zip_object=True),
        output_s3_uri=_s3_uri(
            aws.get("output_s3_uri"),
            "aws.output_s3_uri",
            allow_bucket_root=True,
        ),
        instance_type=instance_type,
        training={**serialize_training_config(training), "backbone_mode": "3d"},
        region=str(aws["region"]) if aws.get("region") else None,
        profile=str(aws["profile"]) if aws.get("profile") else None,
        volume_size_gb=volume,
        managed_spot=managed_spot,
        max_runtime_seconds=runtime,
        max_wait_seconds=wait,
        resource_prefix=prefix,
        ecr_repository=str(aws["ecr_repository"]) if aws.get("ecr_repository") else None,
        execution_role_name=str(aws["execution_role_name"]) if aws.get("execution_role_name") else None,
        execution_role_arn=str(aws["execution_role_arn"]) if aws.get("execution_role_arn") else None,
        image_uri=str(aws["image_uri"]) if aws.get("image_uri") else None,
        network_isolation=bool(aws.get("network_isolation", False)),
        kms_key_arn=str(aws["kms_key_arn"]) if aws.get("kms_key_arn") else None,
        vpc=AwsVpcConfig(
            tuple(map(str, vpc.get("subnet_ids", ()))),
            tuple(map(str, vpc.get("security_group_ids", ()))),
        ),
        tags={str(key): str(value) for key, value in tags.items()},
        rebuild_image=bool(cli_config.get("rebuild_image", False)),
    )


__all__ = [
    "AwsVideoAnomalyTrainingConfig",
    "AwsVpcConfig",
    "load_aws_training_config",
    "serialize_training_config",
]
