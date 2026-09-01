from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AwsImageOneClassSubmission:
    job_name: str
    job_arn: str
    run_id: str
    operation: str
    status: str
    region: str
    managed_spot: bool
    image_uri: str
    variant_count: int
    run_s3_uri: str
    output_s3_uri: str
    console_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageOneClassVariantStatus:
    variant_id: str
    backbone_name: str
    drax_fusion_mode: Optional[str]
    status: str
    benchmark_status: str
    completed_epoch: int = 0
    total_epochs: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class AwsImageOneClassStatus:
    job_name: str
    run_id: Optional[str]
    operation: Optional[str]
    status: str
    current_variant: Optional[str]
    completed_variants: int
    total_variants: int
    progress_percent: Optional[float]
    variants: tuple[ImageOneClassVariantStatus, ...]
    run_s3_uri: Optional[str]
    output_s3_uri: Optional[str]
    failure_reason: Optional[str]
    console_url: str

    @property
    def terminal(self) -> bool:
        return self.status in {"Completed", "Failed", "Stopped"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AwsImageOneClassStopResult:
    job_name: str
    operation: Optional[str]
    status: str
    run_s3_uri: Optional[str]
    next_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "AwsImageOneClassStatus",
    "AwsImageOneClassStopResult",
    "AwsImageOneClassSubmission",
    "ImageOneClassVariantStatus",
]
