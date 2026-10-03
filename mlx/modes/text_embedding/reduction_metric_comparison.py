"""Paired retrieval comparisons for a frozen embedding-reduction study."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.experiment_statistics import holm_adjust, paired_mean_test


METRICS = (
    "ndcg@10", "ndcg@100", "recall@10", "recall@100",
    "mrr@10", "mrr@100", "map@10", "map@100",
)


def _read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise MLXUserError(f"Unable to read retrieval metrics from {path}: {exc}") from exc


def _score(row, metric, path):
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise MLXUserError(f"Invalid {metric} in {path}; provide complete benchmark metrics.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise MLXUserError(f"Invalid {metric} in {path}; retrieval scores must be finite and between 0 and 1.")
    return value


class AnalyzeReductionMetricComparison:
    """Compare one 512D autoencoder with PCA, vanilla AE, and full vectors."""

    def __init__(self, comparison_path, output_path, *, margins, seeds=(42, 43, 44, 45, 46),
                 candidate="orthogonal-mse", pca="pca", vanilla="vanilla-mse", alpha=0.05,
                 expected_datasets=None, study_note=""):
        self.comparison = Path(comparison_path).expanduser().resolve()
        self.output = Path(output_path).expanduser().resolve()
        self.margins = margins
        self.seeds = tuple(seeds)
        self.variants = {"orthogonal": candidate, "pca": pca, "vanilla": vanilla}
        self.alpha = alpha
        self.expected_datasets = tuple(expected_datasets) if expected_datasets is not None else None
        self.study_note = study_note

    def execute(self):
        self._validate_plan()
        datasets, values, dataset_rows = self._collect()
        families = {
            "orthogonal_vs_pca": self._family(datasets, values, "orthogonal", "pca"),
            "orthogonal_vs_vanilla": self._family(datasets, values, "orthogonal", "vanilla"),
            "orthogonal_vs_full": self._family(datasets, values, "orthogonal", "full", noninferiority=True),
            "pca_vs_full": self._family(datasets, values, "pca", "full", noninferiority=True),
            "vanilla_vs_full": self._family(datasets, values, "vanilla", "full", noninferiority=True),
        }
        result = {
            "schema_version": 1,
            "datasets": datasets,
            "seeds": self.seeds,
            "metrics": METRICS,
            "margins": self.margins,
            "alpha": self.alpha,
            "variants": self.variants,
            "unit_of_analysis": "dataset; equal weight after averaging training seeds",
            "ci_policy": "95% paired dataset intervals are unadjusted; decisions use Holm-adjusted p values within each eight-metric family",
            "limitations": "Paired t tests assume approximately independent dataset differences. Small or related dataset suites have limited power and external validity. A failed test does not establish inferiority or equivalence.",
            "study_note": self.study_note,
            "families": families,
            "all_metric_superiority_over_pca": all(row["conclusion"] == "superior" for row in families["orthogonal_vs_pca"]),
            "all_metric_noninferiority_to_full": all(row["conclusion"] == "non-inferior" for row in families["orthogonal_vs_full"]),
            "pca_all_metric_noninferiority_to_full": all(row["conclusion"] == "non-inferior" for row in families["pca_vs_full"]),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output / "statistics.json", result)
        write_csv(self.output / "dataset_metrics.csv", dataset_rows)
        (self.output / "report.md").write_text(self._report(result), encoding="utf-8")
        return result

    def _validate_plan(self):
        if (not isinstance(self.margins, dict) or set(self.margins) != set(METRICS)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1
                       for value in self.margins.values())):
            raise MLXUserError("Provide a positive, finite, predeclared noninferiority margin for every retrieval metric.")
        if (len(self.seeds) < 2 or any(type(seed) is not int for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise MLXUserError("Provide at least two distinct integer training seeds.")
        if self.expected_datasets is not None and (
            len(self.expected_datasets) < 2
            or any(not isinstance(name, str) or not name for name in self.expected_datasets)
            or len(set(self.expected_datasets)) != len(self.expected_datasets)
        ):
            raise MLXUserError("Provide at least two distinct frozen dataset names.")
        if not isinstance(self.study_note, str):
            raise MLXUserError("Study note must be text.")
        if (any(not isinstance(name, str) or not name for name in self.variants.values())
                or len(set(self.variants.values())) != 3):
            raise MLXUserError("Orthogonal, PCA, and vanilla variant names must be distinct.")
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise MLXUserError("Statistical alpha must be between zero and one.")
        if self.output == self.comparison or self.output.is_relative_to(self.comparison):
            raise MLXUserError("Write the metric report outside the immutable experiment comparison stage.")

    def _collect(self):
        baseline_path = self.comparison / "baseline_metrics.csv"
        seed_path = self.comparison / "seed_metrics.csv"
        baselines = {}
        for row in _read_rows(baseline_path):
            name = row.get("dataset")
            if not name or name in baselines:
                raise MLXUserError(f"Duplicate or missing dataset in {baseline_path}.")
            baselines[name] = {metric: _score(row, metric, baseline_path) for metric in METRICS}
        if len(baselines) < 2:
            raise MLXUserError("Metric comparison requires at least two datasets with full-vector baselines.")
        if self.expected_datasets is not None and set(baselines) != set(self.expected_datasets):
            raise MLXUserError("Benchmark datasets differ from the frozen metric protocol.")

        grouped = defaultdict(dict)
        required = set(self.variants.values())
        for row in _read_rows(seed_path):
            if row.get("variant") not in required or row.get("dimension") != "512":
                continue
            dataset, variant = row.get("dataset"), row["variant"]
            try:
                seed = int(row["seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MLXUserError(f"Invalid training seed in {seed_path}.") from exc
            if dataset not in baselines or seed not in self.seeds:
                raise MLXUserError(f"Unexpected dataset or seed for {variant} in {seed_path}.")
            key = dataset, variant
            if seed in grouped[key]:
                raise MLXUserError(f"Duplicate {dataset}/{variant}/seed-{seed} in {seed_path}.")
            grouped[key][seed] = {metric: _score(row, metric, seed_path) for metric in METRICS}

        values, dataset_rows = {}, []
        for dataset in sorted(baselines):
            values[dataset, "full"] = baselines[dataset]
            for role, variant in self.variants.items():
                rows = grouped[dataset, variant]
                if set(rows) != set(self.seeds):
                    raise MLXUserError(f"Incomplete {dataset}/{variant} seed metrics; expected {self.seeds}.")
                values[dataset, role] = {
                    metric: sum(rows[seed][metric] for seed in self.seeds) / len(self.seeds)
                    for metric in METRICS
                }
            for metric in METRICS:
                dataset_rows.append({"dataset": dataset, "metric": metric,
                                     "full": values[dataset, "full"][metric],
                                     "orthogonal": values[dataset, "orthogonal"][metric],
                                     "pca": values[dataset, "pca"][metric],
                                     "vanilla": values[dataset, "vanilla"][metric]})
        return sorted(baselines), values, dataset_rows

    def _family(self, datasets, values, candidate, control, *, noninferiority=False):
        rows = []
        for metric in METRICS:
            differences = [values[dataset, candidate][metric] - values[dataset, control][metric]
                           for dataset in datasets]
            margin = self.margins[metric] if noninferiority else 0.0
            test = paired_mean_test(differences, null_mean=-margin, alpha=self.alpha)
            rows.append({"metric": metric, "candidate": candidate, "control": control,
                         "candidate_mean": sum(values[dataset, candidate][metric] for dataset in datasets) / len(datasets),
                         "control_mean": sum(values[dataset, control][metric] for dataset in datasets) / len(datasets),
                         "margin": margin, **test})
        for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
            row["adjusted_p"] = adjusted
            passed = adjusted < self.alpha and not row["degenerate_variance"]
            row["conclusion"] = ("non-inferior" if passed else "non-inferiority not established") if noninferiority else (
                "superior" if passed else "superiority not established")
        return rows

    @staticmethod
    def _report(result):
        lines = ["# Embedding reduction retrieval comparison", "",
                 f"Datasets (n={len(result['datasets'])}): {', '.join(result['datasets'])}", ""]
        if result["study_note"]:
            lines.extend((result["study_note"], ""))
        lines.extend((result["unit_of_analysis"], "", result["ci_policy"], "", result["limitations"], "",
                      "All eight metrics must establish orthogonal superiority over PCA for the joint superiority claim.",
                      "Training loss, reconstruction, and runtime do not enter these tests.", ""))
        for key, title in (("orthogonal_vs_pca", "Orthogonal versus PCA"),
                           ("orthogonal_vs_vanilla", "Orthogonal versus vanilla AE"),
                           ("orthogonal_vs_full", "Orthogonal versus full Gemma"),
                           ("pca_vs_full", "PCA versus full Gemma"),
                           ("vanilla_vs_full", "Vanilla AE versus full Gemma")):
            lines.extend((f"## {title}", "", "| Metric | Candidate | Control | Mean delta | 95% CI | Margin | Holm p | Conclusion |",
                          "|---|---:|---:|---:|---|---:|---:|---|"))
            for row in result["families"][key]:
                ci = (f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]"
                      if row["ci_low"] is not None else "undefined")
                lines.append(f"| {row['metric']} | {row['candidate_mean']:.6f} | {row['control_mean']:.6f} | "
                             f"{row['mean_difference']:+.6f} | {ci} | {row['margin']:.6f} | "
                             f"{row['adjusted_p']:.6g} | {row['conclusion']} |")
            lines.append("")
        lines.extend((f"All-metric superiority over PCA: {result['all_metric_superiority_over_pca']}", "",
                      f"Orthogonal all-metric noninferiority to full Gemma: {result['all_metric_noninferiority_to_full']}",
                      f"PCA all-metric noninferiority to full Gemma: {result['pca_all_metric_noninferiority_to_full']}", ""))
        return "\n".join(lines)
