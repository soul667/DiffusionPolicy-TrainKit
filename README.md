# Portable Diffusion Policy / LeRobot training container

Designed for one immutable image that can be pulled onto multiple NVIDIA GPU servers.
Code, datasets, checkpoints, Hugging Face cache, and Torch cache are bind-mounted at runtime.

## 0. Host requirement

Each server needs Docker + NVIDIA Container Toolkit and a sufficiently recent NVIDIA driver.
For this CUDA 12.8 / LeRobot 0.6.1 image, use NVIDIA driver >= 570.86.

Test GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## 1. Build once

```bash
docker build -t dp-train:cu128-lr061 .
```

For multiple servers, build once and push to your registry:

```bash
REGISTRY=ghcr.io/YOUR_NAME
docker tag dp-train:cu128-lr061 $REGISTRY/dp-train:cu128-lr061
docker push $REGISTRY/dp-train:cu128-lr061
```

Then on every training server:

```bash
docker pull ghcr.io/YOUR_NAME/dp-train:cu128-lr061
export IMAGE=ghcr.io/YOUR_NAME/dp-train:cu128-lr061
```

## 2. Check the runtime

```bash
./run.sh python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 3. Use the same image for independent jobs on many servers

Mount local code/data/output paths without rebuilding:

```bash
export IMAGE=dp-train:cu128-lr061
export CODE_DIR=/data2/axgu/code/my_policy
export DATA_DIR=/data2/axgu/datasets
export OUTPUT_DIR=/data2/axgu/outputs/dp_exp01

./run.sh bash
```

Or launch a command directly:

```bash
./run.sh python train.py \
  --data /workspace/data/my_dataset.zarr \
  --output /workspace/outputs/exp01
```

For a local package under CODE_DIR:

```bash
./run.sh uv pip install -e /workspace/code
```

Normally install the editable package once per long-lived container/session only if your project needs it.
A cleaner alternative is to put project dependencies into this Dockerfile and rebuild the image.

## 4. Single-server multi-GPU

The image includes `dist-run`, a small `torchrun` wrapper:

```bash
NPROC_PER_NODE=4 ./run.sh dist-run train.py \
  --data /workspace/data/my_dataset.zarr \
  --output /workspace/outputs/exp01
```

For LeRobot's console entry point, this also works:

```bash
NPROC_PER_NODE=4 ./run.sh dist-run lerobot-train \
  --policy.type=diffusion \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=10 \
  --policy.device=cuda \
  --output_dir=/workspace/outputs/dp_ddim10
```

If your LeRobot version/config prefers `accelerate launch`, use that directly instead of `dist-run`.

## 5. Multi-server DDP (optional)

Assume:
- node 0 IP: `10.0.0.10`
- 2 nodes
- 4 GPUs per node
- TCP port 29500 reachable between nodes

Node 0:

```bash
export NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 NPROC_PER_NODE=4
./run.sh dist-run train.py --output /workspace/outputs/exp_multinode
```

Node 1:

```bash
export NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 NPROC_PER_NODE=4
./run.sh dist-run train.py --output /workspace/outputs/exp_multinode
```

`run.sh` uses `--network=host` so the rendezvous address is directly reachable. Do not hard-code
`NCCL_SOCKET_IFNAME` unless NCCL chooses the wrong NIC. If needed:

```bash
export NCCL_SOCKET_IFNAME=eno1
```

For InfiniBand/RDMA machines, keep the host's NVIDIA/IB stack correctly configured and let NCCL auto-detect it first.

## 6. Recommended data layout

```text
host
├── code/        -> /workspace/code
├── datasets/    -> /workspace/data
├── outputs/     -> /workspace/outputs
└── ~/.cache/
    ├── huggingface -> /cache/huggingface
    └── torch       -> /cache/torch
```

This keeps the image immutable and small. Zarr/HDF5/video data and checkpoints never enter the Docker build context.

## 7. Why this layout is useful across servers

- exact same PyTorch / CUDA / LeRobot dependencies everywhere
- code can still be edited on the host without rebuilding the image
- datasets and checkpoints remain on local high-speed disks
- Hugging Face / Torch caches persist across container restarts
- output files are owned by your host UID because `run.sh` uses `--user $(id -u):$(id -g)`
- same wrapper supports 1 GPU, single-node multi-GPU, and optional multi-node `torchrun`
