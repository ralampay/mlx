#!/usr/bin/env python3
"""Run a prepared local remote-sensing adapter pilot through experiment-job.py."""
import argparse
from mlx.modes.object_detection.remote_sensing_pilot import RunRemoteAdapterPilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    RunRemoteAdapterPilot(args.config,resume=args.resume).execute()


if __name__ == "__main__":
    main()
