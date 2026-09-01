from __future__ import annotations

from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.image_recognition_oc.aws.commands import (
    GetAwsImageOneClassStatus,
    ResumeAwsImageOneClass,
    StopAwsImageOneClass,
    SubmitAwsImageOneClass,
    WatchAwsImageOneClass,
)
from mlx.modes.image_recognition_oc.aws.config import load_aws_config
from mlx.modes.image_recognition_oc.aws.presentation import render_aws_result
from mlx.modes.image_recognition_oc.aws.service import SageMakerImageOneClassService


def run_aws_image_one_class(config: dict[str, Any]) -> Any:
    config_path = config.get("config_path")
    if not config_path:
        raise MLXUserError("AWS one-class image actions require --config pointing to YAML.")
    service = SageMakerImageOneClassService(load_aws_config(str(config_path), config))
    action = str(config.get("action") or "train")
    output_format = str(config.get("output_format") or "table")
    if action in {"train", "train-all", "benchmark"}:
        result = SubmitAwsImageOneClass(service).execute()
    elif action == "resume":
        result = ResumeAwsImageOneClass(service, _job(config, action)).execute()
    elif action == "status":
        job_name = _job(config, action)
        if config.get("watch"):
            return WatchAwsImageOneClass(
                service,
                job_name,
                poll_interval=float(config.get("poll_interval", 30)),
                on_status=lambda value: render_aws_result(value, output_format=output_format),
            ).execute()
        result = GetAwsImageOneClassStatus(service, job_name).execute()
    elif action == "stop":
        result = StopAwsImageOneClass(
            service,
            _job(config, action),
            config_path=str(config_path),
        ).execute()
    else:
        raise MLXUserError(
            f"Unsupported AWS one-class image action '{action}'. "
            "Available actions: benchmark, resume, status, stop, train, train-all."
        )
    render_aws_result(result, output_format=output_format)
    return result


def _job(config: dict[str, Any], action: str) -> str:
    value = str(config.get("job_name") or "").strip()
    if not value:
        raise MLXUserError(f"AWS action '{action}' requires --job-name.")
    return value


__all__ = ["run_aws_image_one_class"]
