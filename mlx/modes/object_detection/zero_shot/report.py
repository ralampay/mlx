"""Complete seed comparisons and non-cherry-picked transfer-study reporting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import html

import numpy as np
from scipy.stats import t

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.core.paired_superiority import AnalyzePairedSuperiority, holm_adjust
from .data import read_json


class GenerateTransferReport:
    def __init__(self, output, gallery=None):
        self.output, self.gallery = Path(output).expanduser().resolve(), gallery

    def execute(self):
        root = self.output
        models = read_json(root / "models/manifest.json")["models"]
        datasets = read_json(root / "datasets.json")
        rows, slices = [], []
        for dataset in datasets:
            for model in models:
                directory = root / "runs" / dataset["name"] / model["id"]
                receipt = read_json(directory / "completed.json")
                for name, digest in receipt["artifacts"].items():
                    if sha256_file(directory / name) != digest:
                        raise MLXUserError(
                            f"Changed completed artifact: {directory/name}"
                        )
                rows.append(read_json(directory / "metrics.json"))
                for name, metrics in read_json(directory / "scores.json").items():
                    slices.append(
                        {
                            "dataset": dataset["name"],
                            "method": model["method"],
                            "seed": model["seed"],
                            "slice": name,
                            **metrics,
                        }
                    )
        aggregate = root / "aggregate"
        write_csv(aggregate / "results.csv", rows)
        write_json_atomic(aggregate / "results.json", rows)
        write_csv(aggregate / "slices.csv", slices)
        summaries = self._summaries(rows)
        comparisons = self._comparisons(slices)
        write_csv(aggregate / "summary.csv", summaries)
        write_csv(aggregate / "paired-comparisons.csv", comparisons)
        write_json_atomic(aggregate / "paired-comparisons.json", comparisons)
        self._write_report(summaries, comparisons, datasets)
        if self.gallery:
            self.gallery.execute(datasets, models)
        return {
            "status": "completed",
            "runs": len(rows),
            "images": sum(len(d["images"]) for d in datasets),
            "report": str(aggregate / "summary.md"),
            "gallery": str(root / "gallery/index.html"),
        }

    @staticmethod
    def _summaries(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[(row["dataset"], row["method"])].append(row)
        metrics = (
            "mAP50_95",
            "mAP50",
            "mAP75",
            "APsmall",
            "APmedium",
            "APlarge",
            "AR100",
            "precision",
            "recall",
            "f1",
            "fp_per_image",
            "images_per_second",
            "latency_median_ms",
            "latency_p95_ms",
            "peak_cuda_allocated_mib",
            "peak_cuda_reserved_mib",
            "state_size_mib",
            "trainable_params_at_training",
            "pipeline_seconds",
        )
        output = []
        for (dataset, method), values in sorted(groups.items()):
            row = {"dataset": dataset, "method": method, "seeds": len(values)}
            for key in metrics:
                data = [v[key] for v in values if v[key] is not None]
                mean = float(np.mean(data)) if data else None
                sd = float(np.std(data, ddof=1)) if len(data) > 1 else None
                width = (
                    float(t.ppf(0.975, len(data) - 1)) * sd / math.sqrt(len(data))
                    if sd is not None
                    else None
                )
                row.update(
                    {
                        key: mean,
                        key + "_sd": sd,
                        key + "_ci95_low": mean - width if width is not None else None,
                        key + "_ci95_high": mean + width if width is not None else None,
                    }
                )
            output.append(row)
        return output

    @staticmethod
    def _comparisons(rows):
        groups = defaultdict(dict)
        for row in rows:
            groups[(row["dataset"], row["slice"])].setdefault(row["method"], {})[
                row["seed"]
            ] = row["mAP50_95"]
        result = []
        for (dataset, slice_name), methods in sorted(groups.items()):
            candidate = methods["drax-hybrid"]
            seeds = sorted(candidate)
            for method, baseline in sorted(methods.items()):
                if method == "drax-hybrid":
                    continue
                if method != "frozen" and sorted(baseline) != seeds:
                    raise MLXUserError(f"Unpaired seeds for {dataset}/{method}")
                a, b = [candidate[s] for s in seeds], [
                    baseline[None if method == "frozen" else s] for s in seeds
                ]
                if None in a or None in b:
                    continue
                family = (
                    "primary"
                    if slice_name == "all" and method == "lora"
                    else ("secondary" if slice_name == "all" else "exploratory")
                )
                result.append(
                    {
                        "dataset": dataset,
                        "slice": slice_name,
                        "candidate": "drax-hybrid",
                        "baseline": method,
                        "family": family,
                        **AnalyzePairedSuperiority(a, b).execute(),
                    }
                )
        for family in {r["family"] for r in result}:
            members = [r for r in result if r["family"] == family]
            for row, p in zip(members, holm_adjust([r["p"] for r in members])):
                row["holm_p"] = p
        return result

    def _write_report(self, summaries, comparisons, datasets):
        lines = [
            "# Zero-shot target-domain transfer: YOLOX-L adapters",
            "",
            "No target-domain training or checkpoint selection was performed. All six classes are known to the foundation; zero-shot refers to the target domain, not unseen classes.",
            "",
            "## Complete comparison",
            "",
            "mAP values are percentages. Latency is batch-one GPU forward-only, excluding decode/H2D/NMS. Throughput includes the native inference pipeline; COCO scoring is separate.",
            "",
            "| Dataset | Method | Seeds | mAP50–95 mean ± SD | mAP50 | Small AP | Latency ms | Images/s | Peak allocated MiB | Trainable parameters |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in summaries:
            sd = (
                f" ± {100*r['mAP50_95_sd']:.2f}" if r["mAP50_95_sd"] is not None else ""
            )
            small = f"{100*r['APsmall']:.2f}" if r["APsmall"] is not None else "NA"
            lines.append(
                f"| {r['dataset']} | {r['method']} | {r['seeds']} | {100*r['mAP50_95']:.2f}{sd} | {100*r['mAP50']:.2f} | {small} | {r['latency_median_ms']:.2f} | {r['images_per_second']:.1f} | {r['peak_cuda_allocated_mib']:.0f} | {r['trainable_params_at_training']:.0f} |"
            )
        lines += ["", "## Drax hybrid versus LoRA", ""]
        for row in comparisons:
            if row["family"] != "primary":
                continue
            direction = "higher" if row["mean_difference"] > 0 else "lower"
            lines.append(
                f"- {row['dataset']}: hybrid mAP is {abs(100*row['mean_difference']):.2f} percentage points {direction}; paired 95% CI [{100*row['ci95_low']:.2f}, {100*row['ci95_high']:.2f}], two-sided t-test Holm p={row['holm_p']:.4g}; exact sign-flip p={row['exact_sign_flip_p']:.4g}."
            )
        lines += [
            "",
            "The hybrid's compressed spatial branches may help shifted local appearances and context, whereas LoRA supplies only a low-rank update at these 1×1 convolutions. These results test the combined trained models, not the causal contribution of individual branches. Any accuracy gain must be weighed against the hybrid's greater parameter count, memory, and latency above.",
            "",
            "## Protocol and limitations",
            "",
            "- 640×640 FP32 CUDA; no AMP, TTA, or CUDA graphs. Confidence 0.001, NMS IoU 0.6, post-NMS cap 300; standard COCO AP/AR100. Fixed P/R/F1 and gallery: confidence 0.25, IoU 0.50, COCO category-wise cap 100 and crowd-ignore semantics.",
            "- Model selection uses prior DAWN validation only. Foundation runs once; five adapter seeds are paired by training seed. Mean confidence intervals describe seed variability conditional on these fixed datasets, not generalization to new scenes.",
            "- Two primary hybrid–LoRA tests use Holm correction. Other overall comparisons form a separate secondary family; all slice tests share an exploratory Holm family. Intervals are unadjusted. With five seeds the minimum two-sided exact sign-flip p is 0.0625; non-significance does not demonstrate equivalence.",
            "- MRTMD has only three videos. Per-video and leave-one-video-out rows are sensitivity analyses, not thousands of independent observations; no frame-level significance tests are used.",
            "- ACDC native crowd flags and segmentation-derived areas are retained. MRTMD uses corrected box areas. Object-size metrics are valid within each dataset but not perfectly comparable between their area conventions.",
            "- Validation/test aliases are evaluated only once. Exact-hash provenance checks do not rule out near duplicates or unknown pretraining overlap.",
            "- Latency uses 128 metadata-stratified images × three repetitions, ten warmups; the deterministic interleaved run order reduces but cannot eliminate thermal/background-load effects. Memory is inference memory, not training memory.",
            "- Gallery highlights include improvements, regressions, and near-ties ranked by mean-five-seed image F1. Fixed seed-1 panels are illustrative, never a target-selected best seed. Empty-GT images remain in scoring and the complete gallery.",
            "",
            "## Artifacts",
            "",
            "- `results.csv/json`: every run, metric and timing.",
            "- `summary.csv`: mean, SD and 95% seed intervals.",
            "- `slices.csv`: weather/sequence/video, per-class AP, size AP and leave-one-video-out results.",
            "- `paired-comparisons.csv/json`: complete paired tests and multiplicity families.",
            "- `../gallery/index.html`: all images with comparative boxes and balanced highlights.",
            "- `../models/manifest.json`: relative model paths, hashes and source provenance.",
            "",
        ]
        for dataset in datasets:
            lines.append(
                f"{dataset['name']}: {len(dataset['images'])} images; class counts {dataset['objects_per_class']}; {dataset['crowd_annotations']} crowd annotations. Original data remain at {dataset['images_root']}."
            )
        (self.output / "aggregate/summary.md").write_text("\n".join(lines) + "\n")
        (self.output / "aggregate/summary.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Zero-shot transfer report</title>'
            "<style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:1em}"
            "pre{white-space:pre-wrap;line-height:1.5}a{color:#1670b5}</style>"
            '<p><a href="../gallery/index.html">Complete bounding-box gallery</a> · '
            '<a href="results.csv">Per-run CSV</a> · <a href="paired-comparisons.csv">Paired tests</a></p>'
            "<pre>" + html.escape("\n".join(lines)) + "</pre>"
        )
