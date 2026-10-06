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

Run the tests with `pytest`. The suite runs in parallel with `pytest-xdist`:

```bash
uv run pytest -n auto
```

Tests that train or collect with a real model are marked `slow`. For a quick loop, deselect them:

```bash
uv run pytest -n auto -m "not slow"
```

The first run downloads the Ultralytics datasets and weights the suite uses into `tests/.cache/`, which later runs
reuse. Tests never read or write your global Ultralytics settings.

CI installs the `ci` dependency group on top of `dev` (`uv sync --group ci`), which swaps in CPU-only torch on Linux.
Linux developers without a GPU can do the same to skip the CUDA libraries.

#### Linter and formatter

`ruff` is used for linting and formatting. It is configured in `ruff.toml`. To run the formatter do

```bash
uv run ruff format .
```

To run the linter do

```bash
uv run ruff check .
```

Type checking uses `ty`, configured in `ty.toml`:

```bash
uv run ty check .
```
