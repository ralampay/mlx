from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.requests import TrainImageOneClassRequest


_AWS_KEYS = {
    "region", "profile", "dataset_s3_uri", "output_s3_uri", "instance_type",
    "volume_size_gb", "managed_spot", "max_runtime_seconds", "max_wait_seconds",
    "resource_prefix", "ecr_repository", "execution_role_name", "execution_role_arn",
    "image_uri", "network_isolation", "kms_key_arn", "vpc", "tags",
}
_VPC_KEYS = {"subnet_ids", "security_group_ids"}
_EXCLUDED_TRAINING_KEYS = {
    "model_path", "dataset_path", "dataset_s3_uri", "dataset_cache_dir", "output_path",
    "input_img", "extras",
}
_TRAINING_KEYS = set(TrainImageOneClassRequest.__dataclass_fields__) - _EXCLUDED_TRAINING_KEYS
_BENCHMARK_KEYS = {"enabled", "batch_size", "workers", "plots", "model_s3_uri"}


@dataclass(frozen=True)
class AwsVpcConfig:
    subnet_ids: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AwsImageOneClassConfig:
    action: str
    dataset_s3_uri: str
    output_s3_uri: str
    instance_type: str
    training: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    region: Optional[str] = None
    profile: Optional[str] = None
    volume_size_gb: int = 100
    managed_spot: bool = True
    max_runtime_seconds: int = 86400
    max_wait_seconds: Optional[int] = None
    resource_prefix: str = "mlx-oc"
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


def serialize_training_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(_TRAINING_KEYS) if key in value}


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


def _s3_uri(value: Any, label: str, *, zip_object: bool = False, pth_object: bool = False) -> str:
    normalized = str(value or "").rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise MLXUserError(f"'{label}' must be a valid s3://bucket/key URI.")
    if zip_object and (not parsed.path.strip("/") or not normalized.lower().endswith(".zip")):
        raise MLXUserError(f"'{label}' must point to a .zip dataset object.")
    if pth_object and (not parsed.path.strip("/") or not normalized.lower().endswith(".pth")):
        raise MLXUserError(f"'{label}' must point to a .pth checkpoint object.")
    return normalized


def load_aws_config(config_path: str, cli_config: Mapping[str, Any]) -> AwsImageOneClassConfig:
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
    _reject_unknown(root, {"version", "aws", "training", "benchmark"}, "root")
    if root.get("version", 1) != 1:
        raise MLXUserError("Unsupported AWS configuration version; expected version: 1.")
    aws = dict(_mapping(root.get("aws"), "aws"))
    training = dict(_mapping(root.get("training"), "training"))
    benchmark = dict(_mapping(root.get("benchmark"), "benchmark"))
    _reject_unknown(aws, _AWS_KEYS, "aws")
    _reject_unknown(training, _TRAINING_KEYS, "training")
    _reject_unknown(benchmark, _BENCHMARK_KEYS, "benchmark")
    vpc = _mapping(aws.get("vpc"), "aws.vpc")
    tags = _mapping(aws.get("tags"), "aws.tags")
    _reject_unknown(vpc, _VPC_KEYS, "aws.vpc")

    action = str(cli_config.get("action") or "train")
    explicit = set(cli_config.get("_explicit_options") or ())
    configured_training = dict(training)
    for key in _TRAINING_KEYS & explicit:
        if key in cli_config:
            training[key] = cli_config[key]
    for key in ("instance_type", "profile", "dataset_s3_uri"):
        if key in explicit:
            aws[key] = cli_config.get(key)

    defaults = TrainImageOneClassRequest().to_config()
    training = {**serialize_training_config(defaults), **training, "device": "auto"}
    if training.get("model") != "deep-svdd":
        raise MLXUserError("AWS one-class recognition currently requires training.model: deep-svdd.")
    if (
        action == "train"
        and "backbone" not in configured_training
        and "backbone" not in explicit
    ):
        raise MLXUserError("AWS action 'train' requires training.backbone.")
    if action == "train-all":
        if "backbone" in _mapping(root.get("training"), "training") or "backbone" in explicit:
            raise MLXUserError("AWS train-all selects every compatible backbone; omit training.backbone and --backbone.")
        training.pop("backbone", None)
    probe = TrainImageOneClassRequest.from_config({**training, "backbone": training.get("backbone") or "resnet18"})
    if probe.epochs < 1 or probe.batch_size < 1:
        raise MLXUserError("training.epochs and training.batch_size must be at least 1.")
    if probe.height < 1 or probe.width < 1 or probe.workers < 0:
        raise MLXUserError("training.height/width must be positive and training.workers cannot be negative.")
    if probe.lr is None or probe.lr <= 0:
        raise MLXUserError("training.lr must be greater than zero.")
    if probe.svdd_dim < 1 or probe.svdd_hidden_dim < 1 or not 0 < probe.svdd_quantile < 1:
        raise MLXUserError("SVDD dimensions must be positive and svdd_quantile must be between zero and one.")

    benchmark = {
        "enabled": False,
        "batch_size": int(training.get("batch_size", 16)),
        "workers": int(training.get("workers", 0)),
        "plots": True,
        **benchmark,
    }
    if not isinstance(benchmark["enabled"], bool) or not isinstance(benchmark["plots"], bool):
        raise MLXUserError("benchmark.enabled and benchmark.plots must be YAML booleans.")
    if int(benchmark["batch_size"]) < 1 or int(benchmark["workers"]) < 0:
        raise MLXUserError("benchmark.batch_size must be positive and benchmark.workers cannot be negative.")
    if action == "benchmark":
        benchmark["model_s3_uri"] = _s3_uri(
            benchmark.get("model_s3_uri"), "benchmark.model_s3_uri", pth_object=True
        )
    elif action in {"train", "train-all"} and benchmark.get("model_s3_uri") is not None:
        raise MLXUserError(
            "benchmark.model_s3_uri is used only by standalone AWS benchmark jobs."
        )
    elif benchmark.get("model_s3_uri") is not None:
        benchmark["model_s3_uri"] = _s3_uri(
            benchmark["model_s3_uri"], "benchmark.model_s3_uri", pth_object=True
        )

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
    wait = int(aws["max_wait_seconds"]) if aws.get("max_wait_seconds") is not None else None
    if managed_spot and (wait or min(runtime * 2, 30 * 24 * 60 * 60)) < runtime:
        raise MLXUserError("aws.max_wait_seconds must be at least max_runtime_seconds.")
    prefix = str(aws.get("resource_prefix") or "mlx-oc").strip().lower()
    if not prefix or len(prefix) > 40:
        raise MLXUserError("aws.resource_prefix must contain 1-40 characters.")

    return AwsImageOneClassConfig(
        action=action,
        dataset_s3_uri=_s3_uri(aws.get("dataset_s3_uri"), "aws.dataset_s3_uri", zip_object=True),
        output_s3_uri=_s3_uri(aws.get("output_s3_uri"), "aws.output_s3_uri"),
        instance_type=instance_type,
        training=serialize_training_config(training),
        benchmark=benchmark,
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


__all__ = ["AwsImageOneClassConfig", "AwsVpcConfig", "load_aws_config", "serialize_training_config"]
