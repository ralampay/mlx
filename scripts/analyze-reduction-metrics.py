#!/usr/bin/env python3
"""Analyze a frozen eight-metric reduction protocol after benchmark completion."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx.modes.text_embedding.reduction_metric_comparison import AnalyzeReductionMetricComparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = AnalyzeReductionMetricComparison(
        args.comparison, args.output, margins=protocol["margins"],
        seeds=protocol["seeds"], candidate=protocol["variants"]["orthogonal"],
        pca=protocol["variants"]["pca"], vanilla=protocol["variants"]["vanilla"],
        alpha=protocol["alpha"], expected_datasets=protocol["datasets"],
        study_note=protocol.get("held_out_note", ""),
    ).execute()
    print(json.dumps({"datasets": len(result["datasets"]),
                      "all_metric_superiority_over_pca": result["all_metric_superiority_over_pca"],
                      "all_metric_noninferiority_to_full": result["all_metric_noninferiority_to_full"]}))


if __name__ == "__main__":
    main()
