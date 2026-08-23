#!/usr/bin/env bash
set -euo pipefail

python -m mlx \
    --mode object-detection \
    --platform aws \
    --action status \
    --config ./tmp/aws-training.yaml \
    "$@"
