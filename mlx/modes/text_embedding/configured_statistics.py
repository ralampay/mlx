"""Dataset-level comparisons for named compression experiments."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.experiment_artifacts import read_json
from mlx.modes.text_embedding.experiment_statistics import query_scores, paired_mean_test, holm_adjust


def expected_cells(config):
    return {(dataset, v.name, dimension, seed) for dataset in config.datasets for v in config.variants
            for seed, _, dimensions in v.runs(config.seeds) for dimension in dimensions}


def validate_cells(config, cells):
    keys = [(x["dataset"], x["variant"], x["dimension"], x["seed"]) for x in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected_cells(config):
        raise MLXUserError("Experiment cells are missing, duplicated, or unexpected; complete-suite analysis is unavailable.")


def adjusted_tests(rows, *, noninferiority):
    for row, adjusted in zip(rows, holm_adjust([r["p_value"] for r in rows])):
        row["adjusted_p"] = adjusted
        success = adjusted < row.pop("alpha") and not row["degenerate_variance"]
        row["conclusion"] = ("non-inferior" if success else "non-inferiority not established") if noninferiority else (
            "superior" if success else "superiority not established")
    return rows


@dataclass
class CollectedMetrics:
    baselines: dict
    grouped: dict
    means: dict
    query_rows: list
    seed_rows: list
    training_rows: dict


class AnalyzeConfiguredAutoencoders:
    def __init__(self, request, config, root, cells, output):
        self.request, self.config = request, config
        self.root, self.cells, self.output = Path(root), cells, Path(output)

    def execute(self):
        validate_cells(self.config, self.cells)
        collected = self._collect()
        results = self._compare(collected.means)
        self._write(collected, results)
        return results

    def _collect(self):
        r = self.request
        baselines = {d: query_scores(self.root / d / "baseline/query_metrics.csv", r.primary_metric)
                     for d in self.config.datasets}
        grouped, query_rows, seed_rows, training_rows = {}, [], [], {}
        for cell in self.cells:
            dataset, variant, dimension = cell["dataset"], cell["variant"], cell["dimension"]
            path = self.root / cell["evaluation"]
            baseline = baselines[dataset]
            scores = query_scores(path / "benchmark/query_metrics.csv", r.primary_metric)
            if scores.keys() != baseline.keys():
                raise MLXUserError(f"Mismatched query IDs for {dataset}/{variant}.")
            delta = float(np.mean([scores[q] - baseline[q] for q in baseline]))
            grouped.setdefault((dataset, variant, dimension), []).append(delta)
            query_rows.extend({"dataset": dataset, "variant": variant, "dimension": dimension, "seed": cell["seed"],
                               "query_id": q, "original": baseline[q], "compressed": scores[q],
                               "difference": scores[q] - baseline[q]} for q in baseline)
            metrics = read_json(path / "benchmark/metrics.json")["metrics"]
            seed_rows.append({"dataset": dataset, "variant": variant, "dimension": dimension, "seed": cell["seed"],
                              "training_run": cell["run"] if cell["training"] else "", **metrics,
                              "transform_seconds": read_json(path / "embeddings/stage.json")["seconds"],
                              "retrieval_seconds": read_json(path / "benchmark/stage.json")["seconds"],
                              "vector_bytes": (path / "embeddings/vector_store/vectors.f32").stat().st_size})
            if cell["training"] and cell["run"] not in training_rows:
                training = self.root / cell["training"]
                artifact = training / (f"{cell['kind']}.npz" if cell["kind"] in ("pca", "svd") else "autoencoder.pth")
                training_rows[cell["run"]] = {"dataset": dataset, "variant": variant, "seed": cell["seed"],
                    "training_dimension": cell["training_dimension"], "training_run": cell["run"],
                    "training_seconds": read_json(training / "stage.json")["seconds"], "artifact_bytes": artifact.stat().st_size}
        means = {key: float(np.mean(values)) for key, values in grouped.items()}
        return CollectedMetrics(baselines, grouped, means, query_rows, seed_rows, training_rows)

    def _compare(self, means):
        r = self.request
        combinations = sorted({(v, d) for _, v, d in means})
        primary, difference_tests = [], []

        for variant, dimension in combinations:
            test = paired_mean_test([means[d, variant, dimension] for d in self.config.datasets],
                                    null_mean=-r.noninferiority_margin, alpha=r.alpha)
            primary.append({"variant": variant, "dimension": dimension, "alpha": r.alpha, **test})
            values = [means[d, variant, dimension] for d in self.config.datasets]
            zero_test = paired_mean_test(values, null_mean=0, alpha=r.alpha)
            p = min(1.0, 2 * min(zero_test["p_value"], 1 - zero_test["p_value"])) if not zero_test["degenerate_variance"] else 1.0
            difference_tests.append({"variant": variant, "dimension": dimension, **zero_test, "p_value": p})
        secondary = []
        for candidate, control in self.config.secondary:
            candidate_dims = {d for v, d in combinations if v == candidate}
            control_dims = {d for v, d in combinations if v == control}
            if candidate_dims != control_dims:
                raise MLXUserError("Secondary comparisons require identical evaluation dimensions.")
            for dimension in sorted(candidate_dims):
                values = [means[d, candidate, dimension] - means[d, control, dimension] for d in self.config.datasets]
                secondary.append({"candidate": candidate, "control": control, "dimension": dimension, "alpha": r.alpha,
                                  **paired_mean_test(values, null_mean=0, alpha=r.alpha)})
        for row, adjusted in zip(difference_tests, holm_adjust([x["p_value"] for x in difference_tests])):
            row["adjusted_p"] = adjusted
        results = {"schema_version": 2, "difference_tests": difference_tests,
                   "selection": self.config.selection, "primary_metric": r.primary_metric, "margin": r.noninferiority_margin,
                   "alpha": r.alpha, "primary": adjusted_tests(primary, noninferiority=True),
                   "secondary": adjusted_tests(secondary, noninferiority=False),
                   "unit_of_analysis": "dataset; equal weight after averaging paired queries and seeds",
                   "ci_policy": "unadjusted intervals; Holm-adjusted decisions within each declared family",
                   "limitations": "Transductive corpus adaptation on a mixed full/Nano suite. Selection context is recorded separately. Dataset differences assumed independent and approximately normal. Nonsignificance is not equivalence."}
        return results

    def _write(self, collected, results):
        baselines, grouped, means = collected.baselines, collected.grouped, collected.means
        query_rows, seed_rows, training_rows = collected.query_rows, collected.seed_rows, collected.training_rows
        primary, secondary = results["primary"], results["secondary"]
        baseline_mean = float(np.mean([np.mean(list(scores.values())) for scores in baselines.values()]))
        for row in primary:
            row.update(mean_original=baseline_mean, mean_compressed=baseline_mean + row["mean_difference"])
        self.output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output / "statistics.json", results)
        write_json_atomic(self.output / "cells.json", self.cells)
        write_csv(self.output / "seed_metrics.csv", seed_rows)
        write_csv(self.output / "training_runs.csv", list(training_rows.values()))
        write_csv(self.output / "paired_query_differences.csv", query_rows)
        write_csv(self.output / "dataset_metrics.csv", [
            {"dataset": d, "variant": v, "dimension": width, "mean_difference": means[d, v, width],
             "original": float(np.mean(list(baselines[d].values()))),
             "compressed": float(np.mean(list(baselines[d].values()))) + means[d, v, width],
             "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
             "replicates": len(values), "queries": len(baselines[d])}
            for (d, v, width), values in grouped.items()])
        original_paths = {cell["dataset"]: Path(cell["original"]) for cell in self.cells}
        write_csv(self.output / "baseline_metrics.csv", [
            {"dataset": d, **read_json(self.root / d / "baseline/metrics.json")["metrics"],
             "retrieval_seconds": read_json(self.root / d / "baseline/stage.json")["seconds"],
             "vector_bytes": (original_paths[d] / "vector_store/vectors.f32").stat().st_size,
             "original_embedding_seconds": read_json(original_paths[d] / "stage.json")["seconds"],
             "reused_embeddings": bool(self.request.embedding_source)}
            for d in self.config.datasets])
        lines = ["# Configured autoencoder retrieval experiment", "", results["limitations"], "", results["ci_policy"], "",
                 "| Variant | Dimension | Original | Compressed | Mean delta | 95% CI | Adjusted p | Conclusion |", "|---|---:|---:|---:|---:|---|---:|---|"]
        for x in primary:
            interval = f"[{x['ci_low']:.6f}, {x['ci_high']:.6f}]" if x['ci_low'] is not None else "undefined (zero variance)"
            lines.append(f"| {x['variant']} | {x['dimension']} | {x['mean_original']:.6f} | {x['mean_compressed']:.6f} | {x['mean_difference']:.6f} | {interval} | {x['adjusted_p']:.6g} | {x['conclusion']} |")
        if self.config.selection:
            lines += ["", "## Selection and interpretation", ""]
            lines.extend(f"- {key}: {value}" for key, value in self.config.selection.items())
        lines += ["", "## Two-sided difference tests against original embeddings", "",
                  "| Variant | Dimension | Raw p | Holm p |", "|---|---:|---:|---:|"]
        lines.extend(f"| {x['variant']} | {x['dimension']} | {x['p_value']:.6g} | {x['adjusted_p']:.6g} |" for x in results["difference_tests"])
        lines += ["", "## Prespecified secondary comparisons", ""]
        lines.extend(f"- {x['candidate']} vs {x['control']}, {x['dimension']}D: delta {x['mean_difference']:.6f}; adjusted p {x['adjusted_p']:.6g}; {x['conclusion']}." for x in secondary)
        (self.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
