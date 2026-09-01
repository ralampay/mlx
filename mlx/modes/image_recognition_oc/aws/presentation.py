import json
from typing import Any

from mlx.core.aws.presentation import render_aws_result as _render
from mlx.core.ui import console
from mlx.modes.image_recognition_oc.aws.models import AwsImageOneClassStatus
from rich.table import Table


def render_aws_result(result: Any, *, output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), sort_keys=True))
        return
    if not isinstance(result, AwsImageOneClassStatus):
        _render(result, title="AWS SageMaker One-Class Image Recognition", output_format=output_format)
        return
    summary = Table(title="AWS SageMaker One-Class Image Recognition")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="magenta")
    for label, value in (
        ("Job Name", result.job_name),
        ("Run ID", result.run_id),
        ("Operation", result.operation),
        ("Status", result.status),
        ("Current Variant", result.current_variant),
        ("Variants", f"{result.completed_variants}/{result.total_variants}"),
        ("Progress", f"{result.progress_percent:.2f}%" if result.progress_percent is not None else None),
        ("Run S3 URI", result.run_s3_uri),
        ("Output S3 URI", result.output_s3_uri),
        ("Failure", result.failure_reason),
        ("Console", result.console_url),
    ):
        summary.add_row(label, str(value) if value is not None else "unavailable")
    console.print(summary)
    if result.variants:
        table = Table(title="SVDD Backbone Progress")
        table.add_column("Variant", style="cyan")
        table.add_column("Training")
        table.add_column("Benchmark")
        table.add_column("Epoch", justify="right")
        table.add_column("Error")
        for item in result.variants:
            table.add_row(
                item.variant_id,
                item.status,
                item.benchmark_status,
                f"{item.completed_epoch}/{item.total_epochs}",
                item.error or "",
            )
        console.print(table)


__all__ = ["render_aws_result"]
