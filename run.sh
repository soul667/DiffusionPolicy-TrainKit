#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-dp-train:cu128-lr061}"
CODE_DIR="${CODE_DIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$PWD/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-$HOME/.cache/torch}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$HF_CACHE" "$TORCH_CACHE"

TTY=()
if [[ -t 0 && -t 1 ]]; then
  TTY=(-it)
fi

EXTRA_ENV=()
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  EXTRA_ENV+=( -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" )
fi

exec docker run --rm "${TTY[@]}" \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/container-home \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e NNODES="${NNODES:-1}" \
  -e NODE_RANK="${NODE_RANK:-0}" \
  -e MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
  -e MASTER_PORT="${MASTER_PORT:-29500}" \
  -e NPROC_PER_NODE="${NPROC_PER_NODE:-}" \
  "${EXTRA_ENV[@]}" \
  -v "$CODE_DIR:/workspace/code" \
  -v "$DATA_DIR:/workspace/data" \
  -v "$OUTPUT_DIR:/workspace/outputs" \
  -v "$HF_CACHE:/cache/huggingface" \
  -v "$TORCH_CACHE:/cache/torch" \
  -w /workspace/code \
  "$IMAGE" "$@"
