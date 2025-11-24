# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies into a virtual environment
# We use a virtual environment to isolate dependencies from the system
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-install-project

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf_cache \
    PYTHONUNBUFFERED=1

# Set working directory and ensure ownership
WORKDIR $HOME/app
USER root
RUN chown -R user:user $HOME/app
USER user

# Copy virtual environment from builder
COPY --from=builder --chown=user:user /app/.venv /app/.venv

# Enable the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=user:user . .

# Create cache directory for models
RUN mkdir -p $HF_HOME

# Pre-download AI models during build
RUN python download_models.py

# Expose port 7860
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
