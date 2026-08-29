from __future__ import annotations

from rich.table import Table

from mlx.core.commands import WorkflowEvent
from mlx.core.ui import console, print_info, print_success, print_warning


class RichVideoAnomalyReporter:
    def emit(self, event: WorkflowEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("event") == "video_anomaly_benchmark":
            table = Table(title="Video Anomaly Benchmark")
            table.add_column("Level", style="cyan")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for level, metrics in payload["metrics"].items():
                for key in ("auroc", "auprc", "precision", "recall", "f1", "balanced_accuracy"):
                    table.add_row(level, key, f"{float(metrics[key]):.4f}")
            console.print(table)
            return
        if event.level == "success":
            print_success(event.message)
        elif event.level == "warning":
            print_warning(event.message)
        else:
            print_info(event.message)


def display_video_anomaly_models(summaries) -> None:
    table = Table(title="Video Anomaly Detection Backbones", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("3D class")
    table.add_column("Drax fusion")
    table.add_column("Family")
    table.add_column("Feature dim", justify="right")
    table.add_column("Parameters", justify="right")
    table.add_column("Pretrained")
    table.add_column("Compatible")
    for item in summaries:
        table.add_row(
            item.model_name,
            item.backbone_class,
            item.drax_fusion_mode or "—",
            item.model_family,
            f"{item.feature_dim:,}",
            f"{item.parameter_count:,}",
            "yes" if item.pretrained_available else "no",
            "yes" if item.compatible else "no",
        )
    console.print(table)


__all__ = ["RichVideoAnomalyReporter", "display_video_anomaly_models"]
