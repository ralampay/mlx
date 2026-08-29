from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AwsVideoAnomalyBatchSubmission:
    job_name: str
    job_arn: str
    batch_id: str
    status: str
    region: str
    managed_spot: bool
    image_uri: str
    model_count: int
    batch_s3_uri: str
    output_s3_uri: str
    console_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoAnomalyVariantStatus:
    variant_id: str
    model_name: str
    drax_fusion_mode: Optional[str]
    status: str
    completed_epoch: int = 0
    total_epochs: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class AwsVideoAnomalyBatchStatus:
    job_name: str
    batch_id: Optional[str]
    status: str
    current_variant: Optional[str]
    completed_models: int
    total_models: int
    progress_percent: Optional[float]
    variants: tuple[VideoAnomalyVariantStatus, ...]
    batch_s3_uri: Optional[str]
    output_s3_uri: Optional[str]
    failure_reason: Optional[str]
    console_url: str

    @property
    def terminal(self) -> bool:
        return self.status in {"Completed", "Failed", "Stopped"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "AwsVideoAnomalyBatchStatus",
    "AwsVideoAnomalyBatchSubmission",
    "VideoAnomalyVariantStatus",
]
