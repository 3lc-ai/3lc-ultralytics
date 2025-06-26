#!/bin/bash
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t 3lc-ultralytics-test .

# Run the tests in the Docker container
echo "Running tests in Docker container..."
docker run \
  -v "$(pwd)":/app \
  3lc-ultralytics-test

echo "Tests completed."