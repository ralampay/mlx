from typing import Any

from mlx.core.aws.presentation import render_aws_result as _render


def render_aws_result(result: Any, *, output_format: str = "table") -> None:
    _render(result, title="AWS SageMaker Object Detection", output_format=output_format)
