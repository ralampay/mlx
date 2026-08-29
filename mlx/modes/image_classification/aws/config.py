from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.models import model_family_for
from mlx.modes.image_classification.ood.deep_svdd import validate_svdd_config
from mlx.modes.image_classification.requests import ImageClassificationRequest


_AWS_KEYS = {
    "region", "profile", "dataset_s3_uri", "checkpoint_s3_uri", "instance_type",
    "volume_size_gb", "managed_spot", "max_runtime_seconds", "max_wait_seconds",
    "resource_prefix", "ecr_repository", "execution_role_name", "execution_role_arn",
    "image_uri", "network_isolation", "kms_key_arn", "vpc", "tags",
}
_VPC_KEYS = {"subnet_ids", "security_group_ids"}
_TRAINING_KEYS = {
    "model", "device", "width", "height", "input_size", "batch_size", "epochs", "lr",
    "colored", "pretrained", "embedding_size", "num_pairs", "random_seed", "use_best",
    "verbose", "apply_transformations", "ood_method", "svdd_weight", "svdd_dim",
    "svdd_hidden_dim", "svdd_quantile", "svdd_warmup_epochs", "drax_fusion_mode",
    "draxnet_fusion_mode", "drax_mobilenet_adapter_dim",
    "drax_mobilenet_blocks", "drax_mobilenet_drop_path",
    "drax_mobilenet_efficient_attention", "drax_mobilenet_use_attention",
    "draxnet_stage_blocks",
}
_YAML_TRAINING_KEYS = _TRAINING_KEYS - {"input_size"}
_CLI_TRAINING_KEYS = set(_TRAINING_KEYS)


@dataclass(frozen=True)
class AwsVpcConfig:
    subnet_ids: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AwsTrainingConfig:
    dataset_s3_uri: str
    checkpoint_s3_uri: str
    instance_type: str
    training: ImageClassificationRequest
    region: Optional[str] = None
    profile: Optional[str] = None
    volume_size_gb: int = 100
    managed_spot: bool = True
    max_runtime_seconds: int = 86400
    max_wait_seconds: Optional[int] = None
    resource_prefix: str = "mlx-ic"
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
        return self.max_wait_seconds or min(self.max_runtime_seconds * 2, 30 * 24 * 60 * 60)


def serialize_training_request(request: ImageClassificationRequest) -> dict[str, Any]:
    """Return the stable, mode-owned payload allowed inside a SageMaker run spec."""

    config = request.to_config()
    return {key: config[key] for key in sorted(_TRAINING_KEYS) if key in config}


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


def _s3_uri(value: Any, label: str, *, bucket_root: bool = False) -> str:
    normalized = str(value or "").rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "s3" or not parsed.netloc or (not bucket_root and not parsed.path.strip("/")):
        raise MLXUserError(f"'{label}' must be a valid s3://bucket/{'prefix' if bucket_root else 'key'} URI.")
    if label == "aws.dataset_s3_uri" and not normalized.lower().endswith(".zip"):
        raise MLXUserError("'aws.dataset_s3_uri' must point to a .zip dataset object.")
    return normalized


def load_aws_training_config(config_path: str, cli_config: Mapping[str, Any]) -> AwsTrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise MLXUserError("AWS configuration requires PyYAML. Install MLX with the 'aws' extra.") from exc
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
    _reject_unknown(training, _YAML_TRAINING_KEYS, "training")
    vpc = _mapping(aws.get("vpc"), "aws.vpc")
    tags = _mapping(aws.get("tags"), "aws.tags")
    _reject_unknown(vpc, _VPC_KEYS, "aws.vpc")

    explicit = set(cli_config.get("_explicit_options") or ())
    for key in _CLI_TRAINING_KEYS & explicit:
        if key in cli_config:
            training[key] = cli_config[key]
    if "instance_type" in explicit:
        aws["instance_type"] = cli_config.get("instance_type")
    if "profile" in explicit:
        aws["profile"] = cli_config.get("profile")
    if "dataset_s3_uri" in explicit:
        action = str(cli_config.get("action") or "train")
        if action not in {"train", "resume"}:
            raise MLXUserError(
                "--dataset-s3-uri is supported only for AWS train or resume actions."
            )
        if "dataset_path" in explicit:
            raise MLXUserError("Use either --dataset or --dataset-s3-uri, not both.")
        aws["dataset_s3_uri"] = cli_config.get("dataset_s3_uri")
    training.setdefault("device", "auto")
    if not training.get("model"):
        raise MLXUserError("AWS image-classification training requires 'training.model'.")
    family = model_family_for(str(training["model"]))
    width = int(training.get("width", 224))
    height = int(training.get("height", 224))
    if width < 1 or height < 1:
        raise MLXUserError("training.width and training.height must be at least 1.")
    if int(training.get("epochs", 100)) < 1:
        raise MLXUserError("training.epochs must be at least 1.")
    if int(training.get("batch_size", 1)) < 1:
        raise MLXUserError("training.batch_size must be at least 1.")
    if int(training.get("num_pairs", 100)) < 1:
        raise MLXUserError("training.num_pairs must be at least 1.")
    validate_svdd_config(training)
    if family == "one-shot" and training.get("ood_method", "none") != "none":
        raise MLXUserError(
            "Deep SVDD is supported only for standard image-classification models, "
            "not one-shot models."
        )
    training["width"], training["height"] = width, height
    training["input_size"] = (width, height)

    instance_type = str(aws.get("instance_type") or "").strip()
    if not instance_type:
        raise MLXUserError("AWS configuration requires 'aws.instance_type'.")
    volume = int(aws.get("volume_size_gb", 100))
    runtime = int(aws.get("max_runtime_seconds", 86400))
    if volume < 1:
        raise MLXUserError("aws.volume_size_gb must be at least 1.")
    if runtime < 1 or runtime > 28 * 24 * 60 * 60:
        raise MLXUserError("aws.max_runtime_seconds must be between 1 second and 28 days.")
    managed_spot = bool(aws.get("managed_spot", True))
    wait_value = aws.get("max_wait_seconds")
    wait = int(wait_value) if wait_value is not None else None
    if managed_spot and (wait or min(runtime * 2, 30 * 24 * 60 * 60)) < runtime:
        raise MLXUserError("aws.max_wait_seconds must be greater than or equal to max_runtime_seconds.")
    prefix = str(aws.get("resource_prefix") or "mlx-ic").strip().lower()
    if not prefix or len(prefix) > 40:
        raise MLXUserError("aws.resource_prefix must contain 1-40 characters.")

    return AwsTrainingConfig(
        dataset_s3_uri=_s3_uri(aws.get("dataset_s3_uri"), "aws.dataset_s3_uri"),
        checkpoint_s3_uri=_s3_uri(aws.get("checkpoint_s3_uri"), "aws.checkpoint_s3_uri", bucket_root=True),
        instance_type=instance_type,
        training=ImageClassificationRequest.from_config(training),
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
            subnet_ids=tuple(map(str, vpc.get("subnet_ids", ()))),
            security_group_ids=tuple(map(str, vpc.get("security_group_ids", ()))),
        ),
        tags={str(key): str(value) for key, value in tags.items()},
        rebuild_image=bool(cli_config.get("rebuild_image", False)),
    )


__all__ = [
    "AwsTrainingConfig",
    "AwsVpcConfig",
    "load_aws_training_config",
    "serialize_training_request",
]
