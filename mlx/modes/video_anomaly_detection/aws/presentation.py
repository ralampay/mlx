import json
from typing import Any

from mlx.core.aws.presentation import render_aws_result as _render
from mlx.core.ui import console
from mlx.modes.video_anomaly_detection.aws.models import AwsVideoAnomalyBatchStatus
from rich.table import Table


def render_aws_result(result: Any, *, output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), sort_keys=True))
        return
    if isinstance(result, AwsVideoAnomalyBatchStatus):
        summary = Table(title="AWS SageMaker Video Anomaly Batch Training")
        summary.add_column("Field", style="cyan")
        summary.add_column("Value", style="magenta")
        for label, value in (
            ("Job Name", result.job_name),
            ("Batch ID", result.batch_id),
            ("Status", result.status),
            ("Current Variant", result.current_variant),
            ("Models", f"{result.completed_models}/{result.total_models}"),
            (
                "Progress",
                f"{result.progress_percent:.2f}%"
                if result.progress_percent is not None
                else "unavailable",
            ),
            ("Batch S3 URI", result.batch_s3_uri),
            ("Output S3 URI", result.output_s3_uri),
            ("Failure", result.failure_reason),
            ("Console", result.console_url),
        ):
            summary.add_row(label, str(value) if value is not None else "unavailable")
        console.print(summary)
    else:
        _render(
            result,
            title="AWS SageMaker Video Anomaly Batch Training",
            output_format=output_format,
        )
    if isinstance(result, AwsVideoAnomalyBatchStatus) and result.variants:
        table = Table(title="Video Anomaly Model Progress")
        table.add_column("Variant", style="cyan")
        table.add_column("Status")
        table.add_column("Epoch", justify="right")
        table.add_column("Error")
        for item in result.variants:
            table.add_row(
                item.variant_id,
                item.status,
                f"{item.completed_epoch}/{item.total_epochs}",
                item.error or "",
            )
        console.print(table)


__all__ = ["render_aws_result"]
