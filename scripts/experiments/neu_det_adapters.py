#!/usr/bin/env python3
"""Run a prepared NEU-DET transfer study; suitable for experiment-job.py."""
import argparse
from mlx.modes.object_detection.adapter_transfer import RunQueuedTransferStudy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    RunQueuedTransferStudy(args.config,resume=args.resume).execute()


if __name__ == "__main__":
    main()
