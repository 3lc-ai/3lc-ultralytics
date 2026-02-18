# Contributing

We welcome contributions to this project. Changes should be proposed through GitHub Pull Requests into the `develop` branch.

## Development and Testing

To set up the project locally, follow the steps below.

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

Run the tests with `pytest`.

```bash
uv run pytest
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

