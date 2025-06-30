#!/bin/bash
set -e

if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
    echo "Error: Must be run from repository root folder"
    exit 1
fi

if [ -z "${TLC_API_KEY}" ]; then
    echo "Error: TLC_API_KEY environment variable must be defined"
    exit 1
fi

if [ -d "venv" ]; then
    echo "Error: venv directory exists. Please remove it before running tests in Docker,"
    echo "since this folder will be mounted in the docker repository."
    exit 1
fi

echo "Building Docker image..."
docker build -t 3lc-ultralytics-test -f docker/Dockerfile .

echo "Running tests in Docker container..."
docker run \
  -e "TLC_API_KEY=${TLC_API_KEY}" \
  -v "$(pwd)":/app/3lc-ultralytics \
  3lc-ultralytics-test test

echo "Tests completed."
