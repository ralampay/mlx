"""Object-detection AWS lifecycle and artifact value types."""

from dataclasses import asdict, dataclass
from typing import Any, Optional

from mlx.core.aws.models import *  # noqa: F401,F403


@dataclass(frozen=True)
class AwsBestModelLocation:
    job_name: str
    run_id: str
    provider: str
    model: str
    completed_epoch: Optional[int]
    best_model_s3_uri: str
    size_bytes: int
    last_modified: Optional[str]
    sha256: Optional[str]
    download_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
