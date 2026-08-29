from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from mlx.modes.image_classification.aws.models import AwsTrainingStatus


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
    hp = description.get("HyperParameters", {})
    transitions = description.get("SecondaryStatusTransitions", [])
    secondary = transitions[-1].get("Status") if transitions else None
    creation = description.get("CreationTime")
    start = description.get("TrainingStartTime")
    end = description.get("TrainingEndTime")
    current = now or datetime.now(timezone.utc)
    epoch_target = total_epochs(hp)
    epoch = latest_metric(description, "mlx:epoch")
    eta = latest_metric(description, "mlx:eta_seconds")
    completed = int(epoch) if epoch is not None else None
    training_seconds = description.get("TrainingTimeInSeconds")
    billable = description.get("BillableTimeInSeconds")
    output = description.get("ModelArtifacts", {}).get(
        "S3ModelArtifacts"
    ) or description.get("OutputDataConfig", {}).get("S3OutputPath")
    eta_seconds = int(eta) if eta is not None else None
    return AwsTrainingStatus(
        job_name=description["TrainingJobName"],
        run_id=hp.get("mlx_run_id"),
        status=description["TrainingJobStatus"],
        secondary_status=secondary,
        completed_epoch=completed,
        total_epochs=epoch_target,
        progress_percent=(
            min(100.0, completed / epoch_target * 100.0)
            if completed is not None and epoch_target
            else None
        ),
        elapsed_seconds=(
            int(((end or current) - creation).total_seconds()) if creation else None
        ),
        training_seconds=training_seconds,
        billable_seconds=billable,
        managed_spot=bool(description.get("EnableManagedSpotTraining", False)),
        spot_savings_percent=(
            max(0.0, min(100.0, (1 - billable / training_seconds) * 100))
            if training_seconds and billable is not None
            else None
        ),
        eta_seconds=eta_seconds,
        expected_completion_time=(
            current + timedelta(seconds=eta_seconds)
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
