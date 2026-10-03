#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MLX_ROOT="${MLX_ROOT:-$(dirname -- "$SCRIPT_DIRECTORY")}"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$SCRIPT_DIRECTORY/experiments/autoencoders-orthogonal-384-512.json}"
export EXPERIMENT_OUTPUT="${EXPERIMENT_OUTPUT:-$HOME/Desktop/experiments/gemma-ae-orthogonal-384-512-v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
for option in "$@"; do
  case "$option" in
    --output|--output=*) echo "Set EXPERIMENT_OUTPUT instead of --output so the job lock matches the destination." >&2; exit 2 ;;
  esac
done
JOB_ACTION=run
if [[ "${1:-}" == "--background" ]]; then
  JOB_ACTION=start
  shift
fi
exec "${PYTHON_BIN:-python}" "$SCRIPT_DIRECTORY/experiment-job.py" "$JOB_ACTION" \
  --output "$EXPERIMENT_OUTPUT" -- bash "$SCRIPT_DIRECTORY/test-autoencoders-v2.sh" "$@"
