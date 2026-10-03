#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIRECTORY/test-autoencoders-orthogonal-384-512.sh" --background "$@"
