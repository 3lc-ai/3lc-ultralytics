#!/bin/bash
set -e

source /app/.venv/bin/activate
cd /app/3lc-ultralytics

# If no command is provided, run the tests
if [ $# -eq 0 ]; then
    echo "Running tests..."
    uv run --no-sources --active pytest -k "test_dataset_determinism[train]" -s
else
    # Otherwise, execute the provided command
    exec "$@"
fi
