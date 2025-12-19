# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder
ARG INSTALL_JELLYFIN_FFMPEG=false

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY pyproject.toml .
COPY uv.lock .
# Copy README.md because it is referenced in pyproject.toml
COPY README.md .

# Install dependencies into a virtual environment
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-install-project

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Create user
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

USER root
RUN if [ "${INSTALL_JELLYFIN_FFMPEG}" = "true" ]; then \
            apt-get update && apt-get install -y --no-install-recommends jellyfin-ffmpeg && rm -rf /var/lib/apt/lists/*; \
        fi
USER user
# Copy virtual environment from builder
COPY --from=builder --chown=user:user /app/.venv /app/.venv

# Enable the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=user:user . .

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
