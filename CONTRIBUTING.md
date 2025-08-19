# Contributing

We welcome contributions to this project. Changes should be proposed through GitHub Pull Requests into the `develop` branch.

## Development and Testing

To set up the project locally, you can work in your environment or use Docker to mirror the CI/CD environment.

### Local Setup

#### Prerequisites

- Git for cloning the repository
- `uv` installed with version that adheres to the one specified in `pyproject.toml`.

#### Initial setup

Clone the repository

```bash
git clone https://github.com/3lc-ai/3lc-ultralytics.git
cd 3lc-ultralytics
```

#### Tests

Run the tests with `pytest`. `pytest-xdist` can be enabled by using `-n auto` for faster execution.

```bash
uv run pytest -n auto
```

#### Linter and formatter

`ruff` is used for linting and formatting. It is configured in `ruff.toml`. To run the formatter do

```bash
uv run ruff format .
```

To run the linter do

```bash
uv run ruff check .
```

### Docker Setup for Testing

This project includes a Docker setup for running tests locally, which mirrors the CI/CD environment. This ensures that tests run consistently across different environments.

#### Prerequisites

- Docker installed on your machine
- Git for cloning the repository

#### Running Tests in Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/3lc-ai/3lc-ultralytics.git
   cd 3lc-ultralytics
   ```

2. Run the tests using the provided script:

   ```bash
   ./docker/run-tests-in-docker.sh
   ```

   This script will:
   - Build a Docker image with all necessary dependencies
   - Mount your local repository inside the Docker container
   - Run the tests inside the container

#### Using Pre-built Docker Images

The CI workflow automatically builds and pushes Docker images to GitHub Container Registry (ghcr.io). You can pull and use these images directly:

```bash
docker pull ghcr.io/3lc-ai/3lc-ultralytics:latest

# Make sure you have defined the tlc api key as the TLC_API_KEY environment variable.
docker run -e "TLC_API_KEY=${TLC_API_KEY}" -v "$(pwd)":/app/3lc-ultralytics ghcr.io/3lc-ai/3lc-ultralytics:latest
```

#### Custom Commands

The docker image is set up to have a "pre baked" virtual env matching the uv lock file. This venv is in /app/.venv
activating it is easy by doing `source /app/.venv/bin/activate`. However, when running uv commands in the repository,
you have to tell uv not to create its own venv. We achieve this by adding `--no-sources --active` to the uv commands.
Also the venv is built for Python 3.9, but the repository defaults to 3.12, so all commands have to also specify python
3.9 by using `-p 3.9`. So to run the tests the command is typically:
```bash
uv run -p 3.9 --no-sources --active pytest
```

If you want to run custom commands inside the Docker container:

```bash
# Using locally built image
docker build -t 3lc-ultralytics-test -f docker/Dockerfile .
docker run -it -v "$(pwd)":/app/3lc-ultralytics 3lc-ultralytics-test /bin/bash

docker run -it -v "$(pwd)":/app/3lc-ultralytics ghcr.io/3lc-ai/3lc-ultralytics:latest /bin/bash
```
