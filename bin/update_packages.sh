#!/usr/bin/env bash

set -euo pipefail

python -m pip list --outdated --format=json \
    | python -c 'import json, sys; print("\n".join(package["name"] for package in json.load(sys.stdin)))' \
    | while IFS= read -r package; do
        python -m pip install --upgrade "$package"
    done

python -m pip install --upgrade --force-reinstall --no-deps \
    "ultralytics @ git+https://github.com/ralampay/ultralytics.git@main"
