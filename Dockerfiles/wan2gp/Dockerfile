# Wan2GP — Video Generator for the GPU Poor
# https://github.com/deepbeepmeep/Wan2GP
#
# CUDA arch targets:
#   8.6  — RTX 3060/3070/3080/3090 (Ampere)
#   8.9  — RTX 4060/4070/4080/4090 (Ada Lovelace)
#   9.0  — H100/H800 (Hopper)

ARG CUDA_VERSION=12.8.1
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG CUDA_ARCHITECTURES="8.9"
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

# SageAttention — compiled for target GPU arch (no GPU needed at build time)
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"
ENV MAX_JOBS="8"

COPY <<'PATCH' /tmp/patch_setup.py
import os
with open('setup.py', 'r') as f:
    content = f.read()
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
arch_set = '{' + ', '.join([f'"{arch}"' for arch in arch_list.split(';')]) + '}'
old_section = '''compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")'''
new_section = 'compute_capabilities = ' + arch_set + '\nprint(f"Manually set compute capabilities: {compute_capabilities}")'
content = content.replace(old_section, new_section)
with open('setup.py', 'w') as f:
    f.write(content)
PATCH

RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention \
    && cd /tmp/sageattention \
    && python3 /tmp/patch_setup.py \
    && pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /tmp/sageattention /tmp/patch_setup.py

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
