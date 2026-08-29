from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.video_anomaly_detection.aws.commands import (
    GetAwsVideoAnomalyBatchStatus,
    ResumeAwsVideoAnomalyBatchTraining,
    SubmitAwsVideoAnomalyBatchTraining,
    WatchAwsVideoAnomalyBatchTraining,
)
from mlx.modes.video_anomaly_detection.aws.config import load_aws_training_config
from mlx.modes.video_anomaly_detection.aws.presentation import render_aws_result
from mlx.modes.video_anomaly_detection.aws.service import SageMakerVideoAnomalyBatchService


def run_aws_video_anomaly_detection(config: dict[str, Any]) -> Any:
    config_path = config.get("config_path")
    if not config_path:
        raise MLXUserError("AWS video-anomaly actions require --config pointing to YAML.")
    service = SageMakerVideoAnomalyBatchService(
        load_aws_training_config(str(config_path), config)
    )
    action = str(config.get("action") or "train-all")
    output_format = str(config.get("output_format") or "table")
    if action == "train-all":
        result = SubmitAwsVideoAnomalyBatchTraining(service).execute()
    elif action == "resume":
        result = ResumeAwsVideoAnomalyBatchTraining(
            service, _job(config, action)
        ).execute()
    elif action == "status":
        job_name = _job(config, action)
        if config.get("watch"):
            return WatchAwsVideoAnomalyBatchTraining(
                service,
                job_name,
                poll_interval=float(config.get("poll_interval", 30)),
                on_status=lambda value: render_aws_result(
                    value, output_format=output_format
                ),
            ).execute()
        result = GetAwsVideoAnomalyBatchStatus(service, job_name).execute()
    else:
        raise MLXUserError(
            f"Unsupported AWS video-anomaly action '{action}'. "
            "Available actions: resume, status, train-all."
        )
    render_aws_result(result, output_format=output_format)
    return result


def _job(config: dict[str, Any], action: str) -> str:
    value = str(config.get("job_name") or "").strip()
    if not value:
        raise MLXUserError(f"AWS action '{action}' requires --job-name.")
    return value


__all__ = ["run_aws_video_anomaly_detection"]
