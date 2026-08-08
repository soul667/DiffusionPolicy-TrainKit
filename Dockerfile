# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG LEROBOT_VERSION=0.6.1

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false

# Runtime + common robotics / image / video dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        openssh-client \
        ffmpeg \
        build-essential \
        pkg-config \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
        htop \
        tmux \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Copy uv from its official container image instead of curl | sh.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Keep the Python environment immutable inside the image.
RUN uv python install ${PYTHON_VERSION} \
    && uv venv /opt/venv --python ${PYTHON_VERSION}

# Pin the CUDA/PyTorch stack explicitly so every server runs the same wheels.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python /opt/venv/bin/python \
      "lerobot[training]==${LEROBOT_VERSION}" \
      wandb zarr numcodecs

# Optional but convenient for local project installs at runtime:
#   uv pip install -e /workspace/code

COPY scripts/container-entrypoint /usr/local/bin/container-entrypoint
COPY scripts/dist-run /usr/local/bin/dist-run
RUN chmod +x /usr/local/bin/container-entrypoint /usr/local/bin/dist-run \
    && mkdir -p /workspace/code /workspace/data /workspace/outputs /cache/huggingface /cache/torch \
    && chmod 0777 /workspace /workspace/code /workspace/data /workspace/outputs /cache /cache/huggingface /cache/torch

WORKDIR /workspace/code

# Build-time sanity check. GPU availability is checked at runtime, not during docker build.
RUN python - <<'PY'
import torch
import lerobot
print("torch:", torch.__version__)
print("torch CUDA wheel:", torch.version.cuda)
print("lerobot:", getattr(lerobot, "__version__", "unknown"))
PY

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/container-entrypoint"]
CMD ["bash"]
