"""Analysis-ready aggregate reports for YOLOX adapter runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_baselines import recorded_baseline_runs, read_artifact


FIELDS = (
    "experiment_id", "method", "seed", "model", "foundation_checkpoint",
    "dataset", "train_images", "val_images", "test_images", "epochs",
    "physical_batch_size", "gradient_accumulation", "effective_batch_size",
    "image_size", "device", "gpu", "amp", "adapter_target",
    "adapter_reduction", "adapter_rank", "trainable_params", "total_params",
    "trainable_percent", "mAP50", "mAP50_95", "precision", "recall",
    "training_seconds", "seconds_per_epoch", "images_per_second",
    "peak_cuda_memory_mb", "checkpoint_size_mb", "inference_latency_ms", "status", "source_run",
)


def _summary(values: list[float]) -> dict:
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    margin = critical * deviation / math.sqrt(len(values)) if len(values) > 1 else None
    return {
        "n": len(values), "mean": average, "standard_deviation": deviation,
        "confidence_interval_95": (
            [average - margin, average + margin] if margin is not None else None
        ),
    }


class GenerateAdapterReport:
    def __init__(self, root: Path, *, baseline_study: Path | None = None, comparison_method: str = "drax", extra_runs: tuple[dict, ...] = ()):
        self.root = Path(root).expanduser().resolve()
        self.baseline_study = baseline_study
        self.comparison_method = comparison_method
        self.extra_runs = extra_runs

    def execute(self) -> dict:
        environment = {}
        environment_path = self.root / "environment.json"
        if environment_path.exists():
            environment = json.loads(environment_path.read_text())
        rows = []
        for path in sorted(self.root.glob("*/seed-*/metrics.json")):
            metrics = json.loads(path.read_text())
            row = {field: metrics.get(field) for field in FIELDS}
            row["gpu"] = environment.get("gpu_name")
            rows.append(row)
        for metrics in self.extra_runs:
            rows.append({**{field: metrics.get(field) for field in FIELDS}, "gpu": environment.get("gpu_name")})
        known = {(row["method"], row["seed"]) for row in rows}
        if len(known) != len(rows):
            raise MLXUserError("Duplicate method/seed in comparison inputs")
        for run in recorded_baseline_runs(self.root, self.baseline_study):
            if (run.method, run.seed) in known:
                raise MLXUserError(f"Duplicate comparison run: {run.method}/seed-{run.seed}")
            row = {field: run.metrics.get(field) for field in FIELDS}
            row["gpu"] = read_artifact(run.directory.parent.parent / "environment.json").get("gpu_name")
            rows.append(row)
        if not rows:
            raise MLXUserError(f"No adapter runs found under {self.root}")

        aggregate = self.root / "aggregate"
        aggregate.mkdir(parents=True, exist_ok=True)
        with (aggregate / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        completed = [row for row in rows if row["status"] == "completed"]
        grouped = {}
        for row in completed:
            grouped.setdefault(row["method"], []).append(row)
        method_summaries = {
            method: {
                metric: _summary([float(row[metric]) for row in values])
                for metric in ("mAP50", "mAP50_95", "precision", "recall")
            }
            for method, values in grouped.items()
        }
        paired = {}
        for comparator in ("bottleneck", "conv-adapter", "full-finetune", "lora", "lora-r100", "convpass", "drax", "frozen"):
            if comparator == self.comparison_method:
                continue
            drax = {row["seed"]: row for row in grouped.get(self.comparison_method, [])}
            other = {row["seed"]: row for row in grouped.get(comparator, [])}
            differences = [
                float(drax[seed]["mAP50_95"]) - float(other[seed]["mAP50_95"])
                for seed in sorted(drax.keys() & other.keys())
            ]
            if differences:
                paired[f"{self.comparison_method}_minus_{comparator}"] = _summary(differences)
        results = {"runs": rows, "method_summaries": method_summaries, "paired_differences": paired}
        (aggregate / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        lines = [
            "# YOLOX-L adapter study", "",
            "Single-seed results are exploratory and do not establish statistical significance.", "",
            "| Method | Seeds | mAP50 | mAP50-95 | Trainable % | Peak CUDA MiB | Train seconds |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, values in sorted(grouped.items()):
            lines.append(
                f"| {method} | {len(values)} | {mean(float(row['mAP50']) for row in values):.4f} | "
                f"{mean(float(row['mAP50_95']) for row in values):.4f} | "
                f"{mean(float(row['trainable_percent']) for row in values):.3f} | "
                f"{mean(float(row['peak_cuda_memory_mb'] or 0) for row in values):.1f} | "
                f"{mean(float(row['training_seconds']) for row in values):.1f} |"
            )
        lines.extend(["", f"## Paired {self.comparison_method} comparisons", ""])
        if not paired:
            lines.append("No paired comparator results are available yet.")
        for name, values in paired.items():
            interval = values["confidence_interval_95"]
            interval_text = (
                f"95% CI [{interval[0]:+.4f}, {interval[1]:+.4f}]"
                if interval else "95% CI unavailable with one seed"
            )
            lines.append(
                f"- {name.replace('_', ' ')}: mean {values['mean']:+.4f}, "
                f"SD {values['standard_deviation']:.4f}, {interval_text}."
            )
        lines.extend([
            "", "Non-significance must not be interpreted as equivalence. A non-inferiority claim requires a prespecified margin and powered design.", "",
        ])
        (aggregate / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return {
            "runs": len(rows),
            "csv": str(aggregate / "results.csv"),
            "json": str(aggregate / "results.json"),
            "summary": str(aggregate / "summary.md"),
        }
