from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.aws.commands import (
    GetAwsImageClassificationTrainingStatus,
    ResumeAwsImageClassificationTraining,
    StopAwsImageClassificationTraining,
    SubmitAwsImageClassificationTraining,
    WatchAwsImageClassificationTraining,
)
from mlx.modes.image_classification.aws.config import load_aws_training_config
from mlx.modes.image_classification.aws.presentation import render_aws_result
from mlx.modes.image_classification.aws.service import SageMakerTrainingService


def run_aws_image_classification(config: dict[str, Any]) -> Any:
    config_path = config.get("config_path")
    if not config_path:
        raise MLXUserError("AWS image-classification actions require --config pointing to YAML.")
    service = SageMakerTrainingService(load_aws_training_config(str(config_path), config))
    action = config.get("action") or "train"
    output_format = str(config.get("output_format") or "table")
    if action == "train":
        result = SubmitAwsImageClassificationTraining(service).execute()
    elif action == "resume":
        result = ResumeAwsImageClassificationTraining(
            service, _job(config, action)
        ).execute()
    elif action == "status":
        job_name = _job(config, action)
        if config.get("watch"):
            return WatchAwsImageClassificationTraining(
                service,
                job_name,
                poll_interval=float(config.get("poll_interval", 30)),
                on_status=lambda value: render_aws_result(
                    value, output_format=output_format
                ),
            ).execute()
        result = GetAwsImageClassificationTrainingStatus(service, job_name).execute()
    elif action == "stop":
        result = StopAwsImageClassificationTraining(
            service,
            _job(config, action),
            config_path=str(config_path),
        ).execute()
    else:
        raise MLXUserError(
            f"Unsupported AWS image-classification action '{action}'. "
            "Available actions: resume, status, stop, train."
        )
    render_aws_result(result, output_format=output_format)
    return result


def _job(config: dict[str, Any], action: str) -> str:
    value = str(config.get("job_name") or "").strip()
    if not value:
        raise MLXUserError(f"AWS action '{action}' requires --job-name.")
    return value
