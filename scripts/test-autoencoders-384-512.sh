#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_OUTPUT="${EXPERIMENT_OUTPUT:-$HOME/Desktop/experiments/gemma-ae-retrieval-384-512-v1}"

# Both losses use the same 512-wide hidden layer for the larger bottlenecks.
exec "$SCRIPT_DIRECTORY/test-autoencoders.sh" \
  --bottleneck-dims 384,512 \
  --hidden-dim 512 \
  "$@"
