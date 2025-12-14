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
    python3-dev \
    ffmpeg \
    libsndfile1

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

# If requested, install OpenVINO into the builder and convert ONNX -> OpenVINO IR
# We run conversion in the builder stage so the final runtime image does NOT
# contain the heavy OpenVINO Python packages. Use pip cache mount to speed up builds.
COPY scripts/convert_model_ci.sh /tmp/convert_model_ci.sh
# Copy minimal files required to download models into builder stage
COPY services/font_manager.py services/
COPY assets assets
COPY download_models.py .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_OPENVINO" = "true" ]; then \
        set -e; \
        echo "⬇️ Pre-downloading models in builder stage..."; \
        /app/.venv/bin/python download_models.py; \
        echo "🚀 Converting ONNX -> OpenVINO IR in builder stage (using pip cache)..."; \
        /bin/uv pip install --python /app/.venv/bin/python openvino openvino-dev onnx; \
        # locate the downloaded ONNX model (search common cache locations), then convert if found \
        MODEL_NAME="UVR_MDXNET_KARA_2.onnx"; \
        echo "🔍 Searching for ${MODEL_NAME}..." && \
        MODEL_PATH=$(find /root /app -name "${MODEL_NAME}" -type f -print -quit 2>/dev/null); \
        if [ -z "$MODEL_PATH" ]; then \
             echo "⚠️ Model not found in specific paths, searching /..." && \
             MODEL_PATH=$(find / -name "${MODEL_NAME}" -type f -print -quit 2>/dev/null); \
        fi; \
        if [ -n "$MODEL_PATH" ]; then \
            echo "✅ Found ONNX model at: ${MODEL_PATH}"; \
            /app/.venv/bin/python -m openvino.tools.mo --input_model "${MODEL_PATH}" --compress_to_fp16 --output_dir /app/models_openvino; \
            echo "✅ Converted artifacts in: /app/models_openvino"; \
        else \
            echo "❌ Could not locate ${MODEL_NAME}; failing build."; \
            exit 1; \
        fi; \
        rm -f /tmp/convert_model_ci.sh; \
    else \
        echo "ℹ️ Skipping OpenVINO conversion in builder"; \
    fi

# Ensure models_openvino exists so runtime-stage COPY won't fail if conversion skipped
RUN mkdir -p /app/models_openvino

# Stage 2: Runtime
FROM python:3.12-slim-bookworm
ARG USE_OPENVINO=false

# Enable apt caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    # Install dependencies for adding repo
    apt-get install -y curl gnupg ca-certificates && \
    # Add Jellyfin repo (Gold standard for FFmpeg HW support)
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://repo.jellyfin.org/jellyfin_team.gpg.key | gpg --dearmor -o /etc/apt/keyrings/jellyfin.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/jellyfin.gpg] https://repo.jellyfin.org/debian bookworm main" > /etc/apt/sources.list.d/jellyfin.list && \
    apt-get update && \
    if [ "$USE_OPENVINO" = "true" ]; then \
        echo "🔧 Installing VA drivers and Jellyfin FFmpeg for OpenVINO/QSV..." && \
        apt-get install -y \
            jellyfin-ffmpeg6 \
            libsndfile1 \
            vainfo \
            libva2 \
            libva-drm2 \
            libdrm2 \
            intel-media-va-driver \
            i965-va-driver \
            libmfx1 \
            ocl-icd-libopencl1 && \
        # Symlink Jellyfin FFmpeg to standard path
        ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
        ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe; \
    else \
        echo "ℹ️ Installing runtime minimal packages (no OpenVINO)..." && \
        apt-get install -y \
            ffmpeg \
            libsndfile1 \
            libva2 \
            libva-drm2 \
            libdrm2 \
            ocl-icd-libopencl1; \
    fi

# Set up non-root user for Hugging Face Spaces
# Add to video and render groups for hardware access
RUN useradd -m -u 1000 user && \
    usermod -aG video,render user || true
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONFAULTHANDLER=1 \
    ORT_DISABLE_TELEMETRY=1 \
    SKIP_AUDIO_SEPARATION=false \
    AUDIO_SEPARATION_TIMEOUT=300 \
    AUDIO_SEPARATOR_MODEL_DIR=/home/user/models

# Create model directory and ensure ownership
RUN mkdir -p $AUDIO_SEPARATOR_MODEL_DIR && \
    chown -R user:user $AUDIO_SEPARATOR_MODEL_DIR
# If you use QSV at runtime, set LIBVA_DRIVER_NAME to the appropriate driver (iHD or i965)
# Example: -e LIBVA_DRIVER_NAME=iHD

# Audio Separation Model Configuration:
# - UVR_MDXNET_KARA_2.onnx (default, lightest/fastest, best for J3455/low-end hardware)
# - UVR-MDX-NET-Inst_HQ_3.onnx (slower, higher quality)
# - UVR-MDX-NET-Inst_Main.onnx (slowest, but high quality)

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
RUN python download_models.py && \
    chown -R user:user $AUDIO_SEPARATOR_MODEL_DIR

# Copy converted OpenVINO models (if produced in builder)
COPY --from=builder --chown=user:user /app/models_openvino /app/models_openvino

# Copy application code (everything else)
COPY --chown=user:user . .

# Expose port 7860
EXPOSE 7860

# Run the application with unbuffered output (-u flag)
CMD ["python", "-u", "app.py"]
