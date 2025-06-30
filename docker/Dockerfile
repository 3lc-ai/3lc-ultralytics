FROM ubuntu:25.04

RUN apt-get update && apt-get install -y curl build-essential libglib2.0-0t64 libgl1

ENV PATH="/root/.local/bin:${PATH}"
ENV UV_MANAGED_PYTHON=1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv -p 3.9
RUN uv sync -p 3.9 --no-sources --no-install-project

COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
