from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from rich.table import Table

from mlx.core.aws.models import AwsTrainingStatus
from mlx.core.ui import console


def render_aws_result(result: Any, *, title: str, output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), sort_keys=True))
        return
    table = Table(title=title, show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in result.to_dict().items():
        if key.endswith("_seconds") and value is not None:
            rendered = str(timedelta(seconds=int(value)))
        elif value is None:
            rendered = "unavailable"
        else:
            rendered = str(value)
        table.add_row(key.replace("_", " ").title(), rendered)
    if isinstance(result, AwsTrainingStatus) and result.status in {"Stopped", "Failed"}:
        table.add_row("Resume", "Use --action resume with this --job-name and the same --config.")
    console.print(table)


__all__ = ["render_aws_result"]
