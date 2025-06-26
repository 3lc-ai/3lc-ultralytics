FROM python:3.12-slim

# Install system dependencies

RUN apt-get update && apt-get install -y \
    git build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv==0.7.9

# Set working directory
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv venv 
RUN uv sync --no-sources --no-install-project

# Create an entrypoint script
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
