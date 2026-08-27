#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${1:-}" && "$1" != -* ]]; then
    job_name="$1"
    shift
    set -- --job-name "$job_name" "$@"
fi

python -m mlx \
    --mode object-detection \
    --platform aws \
    --action status \
    --config ./tmp/aws-training.yaml \
    "$@"
