#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$SCRIPT_DIRECTORY/experiments/autoencoders-v2-pilot.json}"
export EXPERIMENT_OUTPUT="${EXPERIMENT_OUTPUT:-$HOME/Desktop/experiments/gemma-ae-v2-pilot}"
exec "$SCRIPT_DIRECTORY/test-autoencoders-v2.sh" "$@"
