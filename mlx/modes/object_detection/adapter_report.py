"""Comparative and paired-seed summaries for adapter runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from mlx.core.exceptions import MLXUserError


FIELDS = ("method", "seed", "map_50", "map_50_95", "precision", "recall",
          "trainable_params", "total_params", "trainable_percent", "training_seconds",
          "peak_memory_bytes", "checkpoint_size_bytes", "inference_latency_ms")


class GenerateAdapterReport:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def execute(self) -> dict:
        runs = []
        for path in sorted(self.root.glob("*/seed-*/metrics.json")):
            metrics = json.loads(path.read_text())
            if all(key in metrics for key in FIELDS):
                runs.append({key: metrics[key] for key in FIELDS})
        if not runs:
            raise MLXUserError(f"No completed adapter runs under {self.root}")
        with (self.root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(runs)
        grouped = {}
        for row in runs:
            grouped.setdefault(row["method"], []).append(row)
        baseline = {r["seed"]: r for r in grouped.get("frozen", [])}
        full = {r["seed"]: r for r in grouped.get("full-finetune", [])}
        lines = ["# YOLOX adapter comparison", "", "Single-seed results are exploratory; they do not establish statistical significance.",
                 "Raw model forward latency excludes image decoding and NMS; peak memory is allocated GPU memory when available.", "",
                 "| Method | Seeds | mAP50 | mAP50-95 | Precision | Recall | Δ vs frozen | Trainable / total | Train s | Peak MiB | Checkpoint MiB | Forward ms |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method, rows in sorted(grouped.items()):
            paired = [r["map_50_95"] - baseline[r["seed"]]["map_50_95"] for r in rows if r["seed"] in baseline]
            delta = f"{mean(paired):+.4f}" if paired else "n/a"
            memory = [r["peak_memory_bytes"] / 2**20 for r in rows if r["peak_memory_bytes"] is not None]
            peak = f"{mean(memory):.1f}" if memory else "n/a"
            lines.append(f"| {method} | {len(rows)} | {mean(r['map_50'] for r in rows):.4f} | "
                         f"{mean(r['map_50_95'] for r in rows):.4f} | "
                         f"{mean(r['precision'] for r in rows):.4f} | {mean(r['recall'] for r in rows):.4f} | "
                         f"{delta} | {int(mean(r['trainable_params'] for r in rows)):,} / "
                         f"{int(mean(r['total_params'] for r in rows)):,} "
                         f"({mean(r['trainable_percent'] for r in rows):.3f}%) | "
                         f"{mean(r['training_seconds'] for r in rows):.1f} | {peak} | ")
            lines[-1] += f"{mean(r['checkpoint_size_bytes'] / 2**20 for r in rows):.2f} | "
            lines[-1] += f"{mean(r['inference_latency_ms'] for r in rows):.2f} |"
        if baseline:
            lines += ["", "## Frozen foundation", "", "The frozen checkpoint is evaluated on the same test split before interpreting adaptation gains."]
        if full:
            lines += ["", "## Gap to full fine-tuning", ""]
            for method in ("bottleneck", "ssf", "lora", "convpass", "conv-adapter", "drax"):
                rows = grouped.get(method, [])
                differences = [r["map_50_95"] - full[r["seed"]]["map_50_95"] for r in rows if r["seed"] in full]
                if differences:
                    lines.append(f"- {method} minus full fine-tuning: {mean(differences):+.4f} mAP50-95 on paired seeds.")
        lines += ["", "## Paired mAP50-95 differences", ""]
        for comparator in ("bottleneck", "conv-adapter", "full-finetune"):
            if "drax" not in grouped or comparator not in grouped:
                continue
            a = {r["seed"]: r["map_50_95"] for r in grouped["drax"]}
            b = {r["seed"]: r["map_50_95"] for r in grouped[comparator]}
            diffs = [a[seed] - b[seed] for seed in sorted(a.keys() & b.keys())]
            if not diffs:
                continue
            sd = stdev(diffs) if len(diffs) > 1 else 0.0
            # Two-sided Student t critical values for small paired samples.
            critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
            ci = (critical.get(len(diffs) - 1, 1.96) * sd / math.sqrt(len(diffs))
                  if len(diffs) > 1 else None)
            lines.append(f"- Drax minus {comparator}: mean {mean(diffs):+.4f}, SD {sd:.4f}, "
                         + (f"approximate paired 95% CI ±{ci:.4f}" if ci is not None else "CI unavailable with one seed") + ".")
        lines += ["", "Compare precision, recall, memory, latency and artifact sizes in comparison.csv.",
                  "A non-inferiority conclusion requires a prespecified margin and adequate paired seeds.", ""]
        (self.root / "report.md").write_text("\n".join(lines))
        return {"runs": len(runs), "csv": str(self.root / "comparison.csv"),
                "report": str(self.root / "report.md")}
