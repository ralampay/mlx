"""Paired RAG answer and evidence comparisons for completed reducer runs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError


VARIANTS = ("full", "orthogonal-mse", "svd", "pca", "vanilla-mse", "truncate")
ANSWER_METRICS = ("supported_exact", "exact_match", "token_f1")
RETRIEVAL_METRICS = ("all_evidence", "evidence_recall")


def _read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise MLXUserError(f"Unable to read RAG results {path}: {exc}") from exc


class AnalyzeRagReduction:
    """Summarize equal-domain means and paired within-domain bootstrap intervals."""

    def __init__(self, model_outputs, output_path, *, answer_margin=0.05, bootstrap_draws=10000):
        self.model_outputs = {name: Path(path).expanduser() for name, path in model_outputs.items()}
        self.output = Path(output_path).expanduser()
        self.answer_margin = answer_margin
        self.bootstrap_draws = bootstrap_draws

    def execute(self):
        if (not self.model_outputs or type(self.answer_margin) not in (int, float)
                or not math.isfinite(self.answer_margin) or not 0 < self.answer_margin < 1
                or type(self.bootstrap_draws) is not int or self.bootstrap_draws < 100):
            raise MLXUserError("Provide model outputs, a valid answer margin, and at least 100 bootstrap draws.")
        means, comparisons, overlaps = [], [], []
        for model, root in sorted(self.model_outputs.items()):
            answers = self._collect(root / "query_results.csv", ANSWER_METRICS)
            retrieval = self._collect(root / "retrieval_results.csv", RETRIEVAL_METRICS)
            if set(answers) != set(retrieval):
                raise MLXUserError(f"RAG answer/retrieval datasets differ for {model}.")
            for dataset in sorted(answers):
                for variant in VARIANTS:
                    means.append({"model": model, "dataset": dataset, "variant": variant,
                                  **{metric: float(np.mean([row[metric] for row in answers[dataset][variant].values()]))
                                     for metric in ANSWER_METRICS},
                                  **{metric: float(np.mean([row[metric] for row in retrieval[dataset][variant].values()]))
                                     for metric in RETRIEVAL_METRICS}})
                baseline = retrieval[dataset]["orthogonal-mse"]
                control = retrieval[dataset]["svd"]
                overlaps.append({"model": model, "dataset": dataset,
                                 "same_top_k_fraction": sum(
                                     baseline[q]["retrieved_ids"] == control[q]["retrieved_ids"]
                                     for q in baseline) / len(baseline)})
            for metric, data in (("supported_exact", answers), ("exact_match", answers),
                                 ("token_f1", answers), ("all_evidence", retrieval),
                                 ("evidence_recall", retrieval)):
                for control in ("full", "svd", "pca", "vanilla-mse", "truncate"):
                    comparison = self._paired(data, "orthogonal-mse", control, metric)
                    comparison.update(model=model, metric=metric, candidate="orthogonal-mse", control=control)
                    if metric == "supported_exact" and control == "full":
                        comparison["decision"] = (
                            "non-inferiority established" if comparison["ci_low"] > -self.answer_margin
                            else "non-inferiority not established"
                        )
                        comparison["margin"] = self.answer_margin
                    elif metric == "supported_exact" and control == "svd":
                        comparison["decision"] = (
                            "superiority established" if comparison["ci_low"] > 0
                            else "superiority not established"
                        )
                        comparison["margin"] = 0
                    else:
                        comparison["decision"] = "descriptive"
                        comparison["margin"] = None
                    comparisons.append(comparison)
        result = {"models": sorted(self.model_outputs), "answer_margin": self.answer_margin,
                  "bootstrap_draws": self.bootstrap_draws,
                  "unit": "equal dataset means; paired questions resampled within each fixed dataset",
                  "limitations": "Bootstrap intervals are conditional on the sampled datasets and questions. A small dataset suite does not support a broad cross-domain superiority claim. Answer exact match and retrieved gold evidence are proxies for correctness and support; no human faithfulness judgments were collected.",
                  "means": means, "comparisons": comparisons, "top_k_overlap": overlaps}
        self.output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output / "statistics.json", result)
        write_csv(self.output / "means.csv", means)
        write_csv(self.output / "comparisons.csv", comparisons)
        (self.output / "report.md").write_text(self._report(result), encoding="utf-8")
        return result

    @staticmethod
    def _collect(path, metrics):
        grouped = defaultdict(lambda: defaultdict(dict))
        for row in _read_rows(path):
            dataset, variant, query = row.get("dataset"), row.get("variant"), row.get("query_id")
            if not dataset or not query or variant not in VARIANTS or query in grouped[dataset][variant]:
                raise MLXUserError(f"Duplicate or invalid RAG result cell in {path}.")
            try:
                values = {metric: float(row[metric]) for metric in metrics}
            except (KeyError, TypeError, ValueError) as exc:
                raise MLXUserError(f"Invalid RAG metric in {path}.") from exc
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
                raise MLXUserError(f"RAG metric outside [0,1] in {path}.")
            grouped[dataset][variant][query] = {**values, "retrieved_ids": row.get("retrieved_ids")}
        if not grouped:
            raise MLXUserError(f"Empty RAG results: {path}.")
        for dataset, variants in grouped.items():
            if set(variants) != set(VARIANTS):
                raise MLXUserError(f"Incomplete RAG variants for {dataset} in {path}.")
            ids = set(variants["full"])
            if any(set(rows) != ids for rows in variants.values()):
                raise MLXUserError(f"Unpaired RAG questions for {dataset} in {path}.")
        return grouped

    def _paired(self, grouped, candidate, control, metric):
        differences = []
        for dataset in sorted(grouped):
            by_query = grouped[dataset]
            differences.append(np.asarray([
                by_query[candidate][query][metric] - by_query[control][query][metric]
                for query in sorted(by_query["full"])
            ], dtype=float))
        observed = float(np.mean([np.mean(group) for group in differences]))
        rng = np.random.default_rng(731)
        draws = np.zeros(self.bootstrap_draws, dtype=float)
        for group in differences:
            sampled = group[rng.integers(0, len(group), size=(self.bootstrap_draws, len(group)))]
            draws += sampled.mean(axis=1) / len(differences)
        low, high = np.quantile(draws, [0.025, 0.975])
        return {"datasets": len(differences), "questions": sum(len(group) for group in differences),
                "mean_difference": observed, "ci_low": float(low), "ci_high": float(high)}

    @staticmethod
    def _report(result):
        lines = ["# RAG embedding reduction evaluation", "", result["unit"], "", result["limitations"], "",
                 "## Prespecified answer-quality decisions", "",
                 "| Embedding | Comparison | Mean supported-correct delta | 95% bootstrap CI | Decision |",
                 "|---|---|---:|---|---|"]
        for row in result["comparisons"]:
            if row["metric"] == "supported_exact" and row["control"] in ("full", "svd"):
                lines.append(f"| {row['model']} | Orthogonal vs {row['control']} | {row['mean_difference']:+.4f} | "
                             f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['decision']} |")
        lines.extend(("", "## Exact top-k overlap with SVD", "",
                      "| Embedding | Dataset | Same retrieved IDs and order |", "|---|---|---:|"))
        for row in result["top_k_overlap"]:
            lines.append(f"| {row['model']} | {row['dataset']} | {row['same_top_k_fraction']:.3f} |")
        lines.extend(("", "The full per-dataset metrics and all secondary paired comparisons are in `means.csv` and `comparisons.csv`.", ""))
        return "\n".join(lines)
