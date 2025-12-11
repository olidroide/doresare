# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Enable apt caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install build dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    python3-dev

# Copy dependency files first for caching
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Install dependencies into a virtual environment
# We use a virtual environment to isolate dependencies from the system
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ARG USE_OPENVINO=false

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_OPENVINO" = "true" ]; then \
    echo "🚀 Building with OpenVINO support..."; \
    uv remove --no-sync onnxruntime || true; \
    uv add --no-sync onnxruntime-openvino; \
    uv sync --no-install-project; \
    else \
    uv sync --frozen --no-install-project; \
    fi

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# Enable apt caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    intel-media-va-driver \
    libmfx1 \
    vainfo \
    ocl-icd-libopencl1

# Set up non-root user for Hugging Face Spaces
# Add to video and render groups for hardware access
RUN useradd -m -u 1000 user && \
    usermod -aG video,render user || true
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONFAULTHANDLER=1 \
    ORT_DISABLE_TELEMETRY=1 \
    SKIP_AUDIO_SEPARATION=false \
    AUDIO_SEPARATION_TIMEOUT=300

# Audio Separation Model Configuration:
# - UVR-MDX-NET-Inst_HQ_3.onnx (default, faster, ~30s, recommended for low-power CPUs)
# - UVR-MDX-NET-Inst_Main.onnx (slower, ~150s on low-power CPUs, higher quality)
# Set SKIP_AUDIO_SEPARATION=true to use original audio (fastest, lower chord accuracy)

# Set working directory and ensure ownership
WORKDIR $HOME/app
USER root
RUN chown -R user:user $HOME/app
USER user

# Copy virtual environment from builder
COPY --from=builder --chown=user:user /app/.venv /app/.venv

# Enable the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create cache directory for models
RUN mkdir -p $HF_HOME

# Optimization: Copy only files needed for model download first
# This ensures that changes to app code don't invalidate the model download cache
COPY --chown=user:user services/font_manager.py services/
COPY --chown=user:user assets assets
COPY --chown=user:user download_models.py .

# Pre-download AI models during build (now cached separately)
RUN python download_models.py

# Copy application code (everything else)
COPY --chown=user:user . .

# Expose port 7860
EXPOSE 7860

# Run the application with unbuffered output (-u flag)
CMD ["python", "-u", "app.py"]
