from __future__ import annotations

from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.presentation import RichInfrastructureEventRenderer
from mlx.core.ui import console, print_info, print_success, print_warning


class RichImageOneClassReporter:
    def __init__(self) -> None:
        self.infrastructure = RichInfrastructureEventRenderer()

    def emit(self, event: WorkflowEvent) -> None:
        if self.infrastructure.handle(event):
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("event") == "image_one_class_training_epoch":
            metrics = payload["metrics"]
            console.print(
                f"[bold cyan]Epoch {event.current}/{event.total}[/bold cyan]  "
                f"train: [magenta]{float(metrics['train_loss']):.6f}[/magenta]  "
                f"validation: [magenta]{float(metrics['val_loss']):.6f}[/magenta]  "
                f"best: [green]{float(payload['best']):.6f}[/green]"
            )
            return
        if payload.get("event") == "image_one_class_benchmark":
            display_benchmark_metrics(payload["metrics"])
            return
        if event.level == "success":
            print_success(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        else:
            print_info(event.message)


def display_models(summaries) -> None:
    table = Table(title="One-Class Image Recognition Models", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Backbone")
    table.add_column("Backbone class")
    table.add_column("Drax fusion")
    table.add_column("Feature dim", justify="right")
    table.add_column("Parameters", justify="right")
    table.add_column("Pretrained")
    for item in summaries:
        table.add_row(
            item.model_name,
            item.backbone_name,
            item.backbone_class,
            item.drax_fusion_mode or "—",
            f"{item.feature_dim:,}",
            f"{item.parameter_count:,}",
            "yes" if item.pretrained_available else "no",
        )
    console.print(table)


def display_inference(result) -> None:
    table = Table(title="One-Class Image Recognition")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Verdict", result.predicted_label)
    table.add_row("Score", f"{result.anomaly_score:.6f}")
    table.add_row("Threshold", f"{result.threshold:.6f}")
    table.add_row("Model", result.model)
    table.add_row("Backbone", result.backbone)
    console.print(table)


def display_benchmark_metrics(metrics) -> None:
    table = Table(title="One-Class Image Benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in ("auroc", "auprc", "precision", "recall", "specificity", "f1", "balanced_accuracy"):
        table.add_row(key, f"{float(metrics[key]):.4f}")
    console.print(table)


__all__ = [
    "RichImageOneClassReporter",
    "display_benchmark_metrics",
    "display_inference",
    "display_models",
]
