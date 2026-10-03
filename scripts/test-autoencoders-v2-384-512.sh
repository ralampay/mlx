#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$HOME/Desktop/experiments/gemma-ae-v2-selection/confirmation.json}"
export EXPERIMENT_OUTPUT="${EXPERIMENT_OUTPUT:-$HOME/Desktop/experiments/gemma-ae-v2-384-512}"
exec "$SCRIPT_DIRECTORY/test-autoencoders-v2.sh" "$@"
