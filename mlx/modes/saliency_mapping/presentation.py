from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.presentation import RichInfrastructureEventRenderer, RichTrainingMetricsRenderer, TrainingMetricSpec
from mlx.core.ui import confirm_action, console, print_info, print_success, print_warning, prompt_int, prompt_text
from mlx.modes.saliency_mapping.requests import BuildSaliencyDatasetRequest


class RichSaliencyReporter:
    def __init__(self) -> None:
        self.infrastructure = RichInfrastructureEventRenderer()
        self.training = RichTrainingMetricsRenderer(
            (
                TrainingMetricSpec("train_loss", "train loss", higher_is_better=False),
                TrainingMetricSpec("val_loss", "val loss", higher_is_better=False),
                TrainingMetricSpec("val_mae", "val MAE", higher_is_better=False),
                TrainingMetricSpec("val_max_f_beta", "max F-beta", higher_is_better=True),
            ),
            event_names=("saliency_epoch",),
            highlights=(("is_best_validation_mae", "best MAE"),),
        )

    def emit(self, event: WorkflowEvent) -> None:
        if self.infrastructure.handle(event) or self.training.handle(event):
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        name = payload.get("event")
        if name == "saliency_benchmark":
            table = Table(title="Saliency Benchmark", show_lines=True)
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for key in ("mae", "max_f_beta", "mean_f_beta", "best_threshold", "precision", "recall", "parameter_count", "images_per_second_forward"):
                table.add_row(key, f"{float(payload['metrics'][key]):.6f}")
            console.print(table)
            return
        if name == "saliency_tensor_output":
            print_success(event.message)
            print_info(f"Output tensor shape: {payload['output_shape']}")
            return
        if event.level == "success":
            print_success(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        else:
            print_info(event.message)


def print_model_table(summaries) -> None:
    table = Table(title="Saliency Mapping Models", show_lines=True)
    for heading in ("Model", "Parameters", "Backbone", "Pretrained", "Groups", "Output"):
        table.add_column(heading, justify="right" if heading == "Parameters" else "left")
    for item in summaries:
        table.add_row(
            item.model_name,
            f"{item.parameter_count:,}",
            item.backbone,
            "yes" if item.pretrained_supported else "no",
            ", ".join(item.groups),
            "1-channel logits",
        )
    console.print(table)


def print_config(model: str | None, config: dict[str, Any]) -> None:
    table = Table(title=f"Configuration for {model or 'checkpoint'} (saliency mapping)")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in config.items():
        if not key.startswith("_"):
            table.add_row(key, str(value))
    console.print(table)


def resolve_dataset_build_request(request: BuildSaliencyDatasetRequest, pair_count: int):
    del pair_count
    resolved = replace(
        request,
        train_count=request.train_count if request.train_count is not None else prompt_int("How many paired samples for TRAIN?"),
        val_count=request.val_count if request.val_count is not None else prompt_int("How many paired samples for VAL?"),
        test_count=request.test_count if request.test_count is not None else prompt_int("How many paired samples for TEST?"),
    )
    output = resolved.output_path or prompt_text("Enter output path for split saliency dataset")
    overwrite = resolved.overwrite
    if Path(output).exists() and not overwrite:
        confirm_action(f"Output directory '{output}' exists. Overwrite?", abort=True)
        overwrite = True
    return replace(resolved, output_path=output, overwrite=overwrite)


__all__ = ["RichSaliencyReporter", "print_config", "print_model_table", "resolve_dataset_build_request"]
