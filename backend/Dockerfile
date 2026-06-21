FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    ca-certificates \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official script
ADD https://astral.sh/uv/install.sh /uv-bootstrap.sh
RUN sh /uv-bootstrap.sh && rm /uv-bootstrap.sh
ENV PATH="/root/.local/bin:${PATH}"

# Set workspace directory
WORKDIR /app

# Copy pyproject.toml, uv.lock, and submodules
COPY pyproject.toml uv.lock ./
COPY sam3/ sam3/

# Sync dependencies using uv (which installs python 3.13 and all dependencies)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --index-strategy unsafe-best-match

# Copy process_assets.py helper
COPY process_assets.py ./

# Copy the backend files
COPY backend/ backend/

# Set environment variables
ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1

# Expose backend port
EXPOSE 8000

# Start FastAPI server
CMD ["uv", "run", "--frozen", "python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
