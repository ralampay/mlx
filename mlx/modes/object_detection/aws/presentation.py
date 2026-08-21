from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from rich.table import Table

from mlx.core.ui import console
from mlx.modes.object_detection.aws.models import AwsTrainingStatus


def _format_duration(value: Any) -> str:
    if value is None:
        return "unavailable"
    return str(timedelta(seconds=int(value)))


def render_aws_result(result: Any, *, output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), sort_keys=True))
        return
    table = Table(title="AWS SageMaker Object Detection", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    values = result.to_dict()
    for key, value in values.items():
        if key.endswith("_seconds"):
            rendered = _format_duration(value)
        elif value is None:
            rendered = "unavailable"
        else:
            rendered = str(value)
        table.add_row(key.replace("_", " ").title(), rendered)
    if isinstance(result, AwsTrainingStatus) and result.status in {"Stopped", "Failed"}:
        table.add_row(
            "Resume",
            "Use --action resume with this --job-name and the same --config.",
        )
    console.print(table)
