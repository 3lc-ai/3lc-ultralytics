#!/bin/bash
set -e

# Store all script arguments
SCRIPT_ARGS="$@"
NUM_ARGS=$#

source /app/.venv/bin/activate
cd /app/3lc-ultralytics

if [ -d ".venv" ]; then
    echo "Error: .venv directory found in current directory. Please remove it."
    exit -1
fi

if [ "$SCRIPT_ARGS" = "lint" ]; then
    uv run -p 3.9 --no-sources --active ruff format --check .
    uv run -p 3.9 --no-sources --active ruff check .
elif [ "$SCRIPT_ARGS" = "test" ]; then
    export PYTHONPATH=$PWD/tests
    uv run -p 3.9 --no-sources --active pytest -n auto
else
    exec $SCRIPT_ARGS
fi
