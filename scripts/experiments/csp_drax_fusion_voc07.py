#!/usr/bin/env python3
"""Run the corrected paired YOLOX-M/CSP-Drax low-data VOC07 study."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx.modes.object_detection.comparison_experiment import (
    AnalyzeCSPDraxComparison,
    CSPDraxComparisonRequest,
    PrepareCSPDraxComparison,
    RunCSPDraxComparison,
)


DEFAULT_DATASET = Path(
    "~/Desktop/experiments/csp-drax-vs-yolox-m-voc07/dataset/prepared"
)
DEFAULT_OUTPUT = Path(
    "~/Desktop/experiments/csp-drax-fusion-vs-yolox-m-voc07"
)
DEFAULT_SOURCE = Path("~/Desktop/object-detection-models/yolox_m.pth")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "run", "analyze"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    request = CSPDraxComparisonRequest(
        dataset=args.dataset,
        output=args.output,
        source_checkpoint=args.source_checkpoint,
        device=args.device,
    )
    if args.action == "prepare":
        PrepareCSPDraxComparison(request).execute()
    elif args.action == "smoke":
        RunCSPDraxComparison(request).execute(smoke=True)
    elif args.action == "run":
        RunCSPDraxComparison(request).execute()
    else:
        AnalyzeCSPDraxComparison(request).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
