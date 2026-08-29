from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from mlx.core.aws.models import AwsTrainingStatus


def total_epochs(hyperparameters: Mapping[str, str]) -> int | None:
    try:
        return int(json.loads(hyperparameters.get("mlx_training", "{}"))["epochs"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def build_training_status(
    description: Mapping[str, Any],
    *,
    latest_metric: Callable[[Mapping[str, Any], str], float | None],
    console_url: str,
    now: datetime | None = None,
) -> AwsTrainingStatus:
    hyperparameters = description.get("HyperParameters", {})
    transitions = description.get("SecondaryStatusTransitions", [])
    secondary = transitions[-1].get("Status") if transitions else None
    creation = description.get("CreationTime")
    start = description.get("TrainingStartTime")
    end = description.get("TrainingEndTime")
    current_time = now or datetime.now(timezone.utc)
    epoch_target = total_epochs(hyperparameters)
    epoch = latest_metric(description, "mlx:epoch")
    eta = latest_metric(description, "mlx:eta_seconds")
    completed_epoch = int(epoch) if epoch is not None else None
    training_seconds = description.get("TrainingTimeInSeconds")
    billable_seconds = description.get("BillableTimeInSeconds")
    output = description.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not output:
        output = description.get("OutputDataConfig", {}).get("S3OutputPath")
    eta_seconds = int(eta) if eta is not None else None
    return AwsTrainingStatus(
        job_name=description["TrainingJobName"],
        run_id=hyperparameters.get("mlx_run_id"),
        status=description["TrainingJobStatus"],
        secondary_status=secondary,
        completed_epoch=completed_epoch,
        total_epochs=epoch_target,
        progress_percent=(
            min(100.0, completed_epoch / epoch_target * 100.0)
            if completed_epoch is not None and epoch_target
            else None
        ),
        elapsed_seconds=(
            int(((end or current_time) - creation).total_seconds()) if creation else None
        ),
        training_seconds=training_seconds,
        billable_seconds=billable_seconds,
        managed_spot=bool(description.get("EnableManagedSpotTraining", False)),
        spot_savings_percent=(
            max(0.0, min(100.0, (1.0 - billable_seconds / training_seconds) * 100.0))
            if training_seconds and billable_seconds is not None
            else None
        ),
        eta_seconds=eta_seconds,
        expected_completion_time=(
            current_time + timedelta(seconds=eta_seconds)
            if eta_seconds is not None and secondary == "Training"
            else None
        ),
        interruptions=sum(item.get("Status") == "Interrupted" for item in transitions),
        checkpoint_s3_uri=description.get("CheckpointConfig", {}).get("S3Uri"),
        output_s3_uri=output,
        failure_reason=description.get("FailureReason"),
        creation_time=creation,
        start_time=start,
        end_time=end,
        console_url=console_url,
    )


__all__ = ["build_training_status", "total_epochs"]
