from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class AwsInfrastructure:
    region: str
    account_id: str
    role_arn: str
    image_uri: str


@dataclass(frozen=True)
class AwsTrainingSubmission:
    job_name: str
    job_arn: str
    run_id: str
    status: str
    region: str
    managed_spot: bool
    image_uri: str
    checkpoint_s3_uri: str
    output_s3_uri: str
    console_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AwsTrainingStatus:
    job_name: str
    run_id: Optional[str]
    status: str
    secondary_status: Optional[str]
    completed_epoch: Optional[int]
    total_epochs: Optional[int]
    progress_percent: Optional[float]
    elapsed_seconds: Optional[int]
    training_seconds: Optional[int]
    billable_seconds: Optional[int]
    managed_spot: bool
    spot_savings_percent: Optional[float]
    eta_seconds: Optional[int]
    expected_completion_time: Optional[datetime]
    interruptions: int
    checkpoint_s3_uri: Optional[str]
    output_s3_uri: Optional[str]
    failure_reason: Optional[str]
    creation_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    console_url: str

    @property
    def terminal(self) -> bool:
        return self.status in {"Completed", "Failed", "Stopped"}

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for name in (
            "creation_time",
            "start_time",
            "end_time",
            "expected_completion_time",
        ):
            value = values[name]
            if value is not None:
                values[name] = value.isoformat()
        return values


@dataclass(frozen=True)
class AwsTrainingStopResult:
    job_name: str
    status: str
    checkpoint_s3_uri: Optional[str]
    resume_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
