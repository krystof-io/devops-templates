# Wan2GP — Video Generator for the GPU Poor
# https://github.com/deepbeepmeep/Wan2GP
#
ARG CUDA_VERSION=12.8.1
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG WAN2GP_COMMIT=79f82257bc33a11f08e53c881dbc7598c9c57f98

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git wget curl cmake ninja-build \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone Wan2GP at pinned commit (update WAN2GP_COMMIT to upgrade)
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /workspace \
    && cd /workspace \
    && git checkout ${WAN2GP_COMMIT} \
    && rm -rf .git

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch — must match the CUDA version in the base image
RUN pip install --no-cache-dir \
        torch==2.10.0+cu128 \
        torchvision==0.25.0+cu128 \
        torchaudio==2.10.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir -r requirements.txt

# SageAttention 2 — built from source for target GPU arch
# 12.0 = RTX PRO 5000 Blackwell (sm_120); 8.9 = Ada (RTX 4070 Ti Super)
ARG CUDA_ARCHITECTURES="12.0"
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"
ENV MAX_JOBS="8"

RUN git clone --branch v2.2.0 --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sageattention \
    && cd /tmp/sageattention \
    && pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /tmp/sageattention

# Non-root user (UID 1000 matches K8s fsGroup)
RUN useradd -u 1000 -ms /bin/bash user \
    && chown -R user:user /workspace \
    && mkdir -p /home/user/.cache \
    && chown -R user:user /home/user/.cache

ENV HOME=/home/user
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/user/.cache/huggingface
ENV TORCH_ALLOW_TF32_CUBLAS=1
ENV TORCH_ALLOW_TF32_CUDNN=1
ENV SDL_AUDIODRIVER=dummy

EXPOSE 7860

USER user

CMD ["python3", "wgp.py", "--listen"]
