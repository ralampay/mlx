#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
requirements_file="${script_dir}/requirements.txt"

if [[ ! -f "${requirements_file}" ]]; then
    echo "Requirements file not found: ${requirements_file}" >&2
    exit 1
fi

mapfile -t git_requirements < <(
    python - "${requirements_file}" <<'PY'
from pathlib import Path
import sys

for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    requirement = raw_line.strip()
    if requirement and not requirement.startswith("#") and "git+" in requirement:
        print(requirement)
PY
)

if (( ${#git_requirements[@]} == 0 )); then
    echo "No Git dependencies found in ${requirements_file}." >&2
    exit 1
fi

for requirement in "${git_requirements[@]}"; do
    echo "Updating ${requirement}"
    python -m pip install --upgrade --force-reinstall "${requirement}"
done
