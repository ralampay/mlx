"""Paired dataset-level inference; repeated seeds never inflate the sample size."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.experiment_artifacts import read_json


def paired_mean_test(differences, *, null_mean, alpha):
    try:
        from scipy import stats
    except ImportError as exc:
        raise MLXUserError("Retrieval statistics require scipy; install MLX with the retrieval-benchmark extra.") from exc
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise MLXUserError("Statistical comparisons require at least two finite dataset-level differences.")
    result = {"datasets": len(values), "mean_difference": float(values.mean()), "null_mean": null_mean}
    if float(values.std(ddof=1)) < 1e-15:
        return {**result, "p_value": 1.0, "ci_low": None, "ci_high": None,
                "lower_one_sided": None, "degenerate_variance": True}
    test = stats.ttest_1samp(values, popmean=null_mean, alternative="greater")
    se = float(stats.sem(values))
    radius = float(stats.t.ppf(1 - alpha / 2, len(values) - 1)) * se
    lower = float(values.mean()) - float(stats.t.ppf(1 - alpha, len(values) - 1)) * se
    return {**result, "p_value": float(test.pvalue), "ci_low": float(values.mean()) - radius,
            "ci_high": float(values.mean()) + radius, "lower_one_sided": lower,
            "degenerate_variance": False}


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise MLXUserError("Holm correction requires finite probabilities.")
    adjusted = np.zeros(len(values))
    previous = 0.0
    for rank, index in enumerate(np.argsort(values, kind="stable")):
        previous = max(previous, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = previous
    return adjusted.tolist()


def variant_name(loss, dimension, seed):
    return f"{loss}-{dimension}/seed-{seed}"


def query_scores(path, metric):
    try:
        with Path(path).open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        scores = {row["query_id"]: float(row[metric]) for row in rows}
        if not scores or len(scores) != len(rows) or not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("empty, duplicate, or non-finite query metrics")
        return scores
    except (OSError, ValueError, KeyError) as exc:
        raise MLXUserError(f"Invalid query metrics in {path}: {exc}") from exc


class AnalyzeAutoencoderRetrieval:
    def __init__(self, request, datasets, root, output):
        self.request = request
        self.datasets = datasets
        self.root = Path(root)
        self.output = Path(output)

    def execute(self):
        means, dataset_rows, query_rows, seed_rows, baseline_rows = self._collect()
        primary, secondary = self._compare(means)
        return self._write(primary, secondary, dataset_rows, query_rows, seed_rows, baseline_rows)

    def _collect(self):
        request = self.request
        means = {}
        dataset_rows, query_rows, seed_rows, baseline_rows = [], [], [], []
        for dataset in self.datasets:
            directory = self.root / dataset
            baseline = query_scores(directory / "baseline/query_metrics.csv", request.primary_metric)
            original_metrics = read_json(directory / "baseline/metrics.json")
            baseline_rows.append({
                "dataset": dataset, **original_metrics["metrics"],
                "embedding_seconds": read_json(directory / "original/stage.json")["seconds"],
                "retrieval_seconds": read_json(directory / "baseline/stage.json")["seconds"],
                "vector_bytes": (directory / "original/vector_store/vectors.f32").stat().st_size,
            })
            for loss in request.losses:
                for dimension in request.bottleneck_dims:
                    differences = []
                    for seed in request.seeds:
                        variant = variant_name(loss, dimension, seed)
                        cell = directory / variant
                        scores = query_scores(cell / "benchmark/query_metrics.csv", request.primary_metric)
                        if scores.keys() != baseline.keys():
                            raise MLXUserError(f"Mismatched evaluation queries in {dataset}/{variant}.")
                        delta = [scores[q] - baseline[q] for q in baseline]
                        differences.append(float(np.mean(delta)))
                        query_rows.extend({"dataset": dataset, "loss": loss, "dimension": dimension,
                                           "seed": seed, "query_id": q, "original": baseline[q],
                                           "compressed": scores[q], "difference": scores[q] - baseline[q]}
                                          for q in baseline)
                        summary = read_json(cell / "benchmark/metrics.json")
                        seed_rows.append({"dataset": dataset, "loss": loss, "dimension": dimension,
                                          "seed": seed, **summary["metrics"],
                                          "training_seconds": read_json(cell / "training/stage.json")["seconds"],
                                          "transform_seconds": read_json(cell / "embeddings/stage.json")["seconds"],
                                          "retrieval_seconds": read_json(cell / "benchmark/stage.json")["seconds"],
                                          "vector_bytes": (cell / "embeddings/vector_store/vectors.f32").stat().st_size})
                    mean = float(np.mean(differences))
                    means[dataset, loss, dimension] = mean
                    dataset_rows.append({"dataset": dataset, "loss": loss, "dimension": dimension,
                                         "original": float(np.mean(list(baseline.values()))),
                                         "mean_difference": mean, "seed_sd": float(np.std(differences, ddof=1)),
                                         "queries": len(baseline), "seeds": len(request.seeds)})
        return means, dataset_rows, query_rows, seed_rows, baseline_rows

    def _compare(self, means):
        request = self.request
        primary = []
        for loss in request.losses:
            for dimension in request.bottleneck_dims:
                differences = [means[d, loss, dimension] for d in self.datasets]
                primary.append({"loss": loss, "dimension": dimension, **paired_mean_test(
                    differences, null_mean=-request.noninferiority_margin, alpha=request.alpha)})
        for row, adjusted in zip(primary, holm_adjust([row["p_value"] for row in primary])):
            row.update(adjusted_p=adjusted, conclusion=("non-inferior" if adjusted < request.alpha else "non-inferiority not established"))
        secondary = []
        if "mse" in request.losses and "mse-similarity" in request.losses:
            for dimension in request.bottleneck_dims:
                differences = [means[d, "mse-similarity", dimension] - means[d, "mse", dimension] for d in self.datasets]
                secondary.append({"dimension": dimension, **paired_mean_test(differences, null_mean=0, alpha=request.alpha)})
            for row, adjusted in zip(secondary, holm_adjust([row["p_value"] for row in secondary])):
                row.update(adjusted_p=adjusted, conclusion=("similarity loss superior" if adjusted < request.alpha else "superiority not established"))
        return primary, secondary

    def _write(self, primary, secondary, dataset_rows, query_rows, seed_rows, baseline_rows):
        request = self.request
        self.output.mkdir(parents=True, exist_ok=True)
        write_csv(self.output / "baseline_metrics.csv", baseline_rows)
        write_csv(self.output / "dataset_metrics.csv", dataset_rows)
        write_csv(self.output / "seed_metrics.csv", seed_rows)
        write_csv(self.output / "paired_query_differences.csv", query_rows)
        results = {"primary_metric": request.primary_metric, "margin": request.noninferiority_margin,
                   "alpha": request.alpha, "primary": primary, "secondary": secondary,
                   "unit_of_analysis": "dataset; equal weighting after averaging queries and training seeds",
                   "ci_policy": "unadjusted two-sided and one-sided intervals; conclusions use Holm-adjusted p values",
                   "limitations": "Selected mixed full/Nano suite; transductive corpus adaptation. Dataset differences assumed independent and approximately normal. Nonsignificance is not equivalence. No claim for every dataset or unseen corpora."}
        write_json_atomic(self.output / "statistics.json", results)
        lines = ["# Autoencoder retrieval experiment", "", results["unit_of_analysis"], "", results["limitations"], "",
                 results["ci_policy"], "", "| Loss | Dimensions | Mean delta | Adjusted p | Conclusion |",
                 "|---|---:|---:|---:|---|"]
        lines.extend(f"| {r['loss']} | {r['dimension']} | {r['mean_difference']:.6f} | {r['adjusted_p']:.6g} | {r['conclusion']} |" for r in primary)
        lines.extend(["", "## Similarity loss versus MSE", ""])
        lines.extend(f"- {r['dimension']} dimensions: delta {r['mean_difference']:.6f}, adjusted p {r['adjusted_p']:.6g}; {r['conclusion']}." for r in secondary)
        (self.output / "report.md").write_text("\n".join(lines) + "\n")
        return results
