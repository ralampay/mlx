#!/usr/bin/env python3
"""Thin local scratch-comparison entrypoint."""
import argparse
import json

from mlx.modes.object_detection.comparison_suite import RunComparisonSuite, read_json
from mlx.modes.object_detection.comparison_report import AnalyzeComparisonSuite
from mlx.modes.object_detection.libreyolo.comparison_backend import LibreYOLOComparisonBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','pilot','run','status','analyze'))
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    command = RunComparisonSuite(read_json(args.config),LibreYOLOComparisonBackend(),
                                 lambda path: AnalyzeComparisonSuite(path).execute())
    print(json.dumps(command.execute(args.action),default=str,indent=2))


if __name__ == '__main__':
    main()
