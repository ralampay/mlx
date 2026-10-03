import csv

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.reduction_metric_comparison import (
    METRICS, AnalyzeReductionMetricComparison,
)


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _fixture(path):
    path.mkdir()
    datasets = [f"dataset-{number}" for number in range(6)]
    _csv(path / "baseline_metrics.csv", [
        {"dataset": dataset, **{metric: 0.8 for metric in METRICS}}
        for dataset in datasets
    ])
    rows = []
    for index, dataset in enumerate(datasets):
        for variant, score in (("orthogonal-mse", 0.8 + index * 0.002),
                               ("pca", 0.65 + index * 0.001),
                               ("vanilla-mse", 0.7 + index * 0.001)):
            for seed in (42, 43):
                rows.append({"dataset": dataset, "variant": variant, "dimension": 512,
                             "seed": seed, **{metric: score for metric in METRICS}})
    _csv(path / "seed_metrics.csv", rows)
    return rows


def test_complete_eight_metric_comparison_uses_dataset_units(tmp_path):
    comparison = tmp_path / "comparison"
    _fixture(comparison)
    result = AnalyzeReductionMetricComparison(
        comparison, tmp_path / "analysis", margins={metric: 0.01 for metric in METRICS},
        seeds=(42, 43),
    ).execute()
    assert result["all_metric_superiority_over_pca"]
    assert result["all_metric_noninferiority_to_full"]
    assert not result["pca_all_metric_noninferiority_to_full"]
    assert len(result["families"]["orthogonal_vs_vanilla"]) == 8
    assert all(row["datasets"] == 6 for row in result["families"]["orthogonal_vs_pca"])
    assert (tmp_path / "analysis/report.md").exists()


def test_missing_seed_is_rejected(tmp_path):
    comparison = tmp_path / "comparison"
    rows = _fixture(comparison)
    _csv(comparison / "seed_metrics.csv", rows[:-1])
    with pytest.raises(MLXUserError, match="Incomplete"):
        AnalyzeReductionMetricComparison(
            comparison, tmp_path / "analysis", margins={metric: 0.01 for metric in METRICS},
            seeds=(42, 43),
        ).execute()


def test_analysis_cannot_write_inside_comparison(tmp_path):
    comparison = tmp_path / "comparison"
    _fixture(comparison)
    with pytest.raises(MLXUserError, match="outside"):
        AnalyzeReductionMetricComparison(
            comparison, comparison / "metrics", margins={metric: 0.01 for metric in METRICS},
            seeds=(42, 43),
        ).execute()


def test_frozen_dataset_names_must_match(tmp_path):
    comparison = tmp_path / "comparison"
    _fixture(comparison)
    with pytest.raises(MLXUserError, match="frozen metric protocol"):
        AnalyzeReductionMetricComparison(
            comparison, tmp_path / "analysis", margins={metric: 0.01 for metric in METRICS},
            seeds=(42, 43), expected_datasets=("other-a", "other-b"),
        ).execute()
