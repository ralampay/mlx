from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.aws.commands import (
    GetAwsObjectDetectionTrainingStatus,
    LocateBestAwsObjectDetectionModel,
    ResumeAwsObjectDetectionTraining,
    StopAwsObjectDetectionTraining,
    SubmitAwsObjectDetectionTraining,
    WatchAwsObjectDetectionTraining,
)
from mlx.modes.object_detection.aws.config import load_aws_training_config
from mlx.modes.object_detection.aws.presentation import render_aws_result
from mlx.modes.object_detection.aws.service import SageMakerTrainingService


def run_aws_object_detection(config: dict[str, Any]) -> Any:
    config_path = config.get("config_path")
    if not config_path:
        raise MLXUserError("AWS object-detection actions require --config pointing to YAML.")
    aws_config = load_aws_training_config(str(config_path), config)
    service = SageMakerTrainingService(aws_config)
    action = config.get("action") or "train"
    output_format = str(config.get("output_format") or "table")

    if action == "train":
        result = SubmitAwsObjectDetectionTraining(service).execute()
    elif action == "best-model":
        result = LocateBestAwsObjectDetectionModel(service).execute()
    elif action == "resume":
        job_name = _require_job_name(config, action)
        result = ResumeAwsObjectDetectionTraining(service, job_name).execute()
    elif action == "status":
        job_name = _require_job_name(config, action)
        if config.get("watch"):
            result = WatchAwsObjectDetectionTraining(
                service,
                job_name,
                poll_interval=float(config.get("poll_interval", 30.0)),
                on_status=lambda status: render_aws_result(
                    status,
                    output_format=output_format,
                ),
            ).execute()
            return result
        result = GetAwsObjectDetectionTrainingStatus(service, job_name).execute()
    elif action == "stop":
        job_name = _require_job_name(config, action)
        result = StopAwsObjectDetectionTraining(
            service,
            job_name,
            config_path=str(config_path),
        ).execute()
    else:
        raise MLXUserError(
            f"Unsupported AWS object-detection action '{action}'. "
            "Available actions: best-model, resume, status, stop, train."
        )

    render_aws_result(result, output_format=output_format)
    return result


def _require_job_name(config: dict[str, Any], action: str) -> str:
    job_name = str(config.get("job_name") or "").strip()
    if not job_name:
        raise MLXUserError(f"AWS action '{action}' requires --job-name.")
    return job_name
