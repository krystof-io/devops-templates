# ── Version pinning ──────────────────────────────────────────────
# CUDA:              https://hub.docker.com/r/nvidia/cuda/tags?name=devel-ubuntu
# whisper.cpp:       https://github.com/ggml-org/whisper.cpp/releases
# stable-diffusion:  https://github.com/leejet/stable-diffusion.cpp/releases
#                    (no semver — uses auto-generated master-<N>-<hash> tags)
# llama-swap:        https://github.com/mostlygeek/llama-swap/pkgs/container/llama-swap
# PyTorch cu128:     https://download.pytorch.org/whl/cu128/torch/
# torchaudio cu128:  https://download.pytorch.org/whl/cu128/torchaudio/
#   Pin both to the same version. Unpinned torch pulls from PyPI (CUDA 13.0)
#   which breaks torchaudio's CUDA version check.

# Stage 1: Build whisper-server with CUDA support
ARG CUDA_VERSION=12.8.1
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS whisper-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG WHISPER_CPP_VERSION=v1.8.4

RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} \
        https://github.com/ggml-org/whisper.cpp.git /whisper.cpp

WORKDIR /whisper.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -t whisper-server -j$(( $(nproc)))

# Stage 2: Build sd-server (stable-diffusion.cpp) with CUDA support
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS sd-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG SD_CPP_VERSION=master-596-90e87bc

RUN git clone --recursive --depth 1 --branch ${SD_CPP_VERSION} \
        https://github.com/leejet/stable-diffusion.cpp.git /stable-diffusion.cpp

WORKDIR /stable-diffusion.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DSD_BUILD_EXAMPLES=ON \
    && cmake --build build --config Release -t sd-server -j$(( $(nproc) ))

# Stage 3: Combine into the llama-swap CUDA image
FROM ghcr-proxy.private.krystof.io/mostlygeek/llama-swap:cuda

COPY --from=whisper-builder /whisper.cpp/build/bin/whisper-server /app/whisper-server
COPY --from=sd-builder /stable-diffusion.cpp/build/bin/sd-server /app/sd-server

# ── Python / system deps for TTS servers ──────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev libpython3.12 \
        build-essential ffmpeg sox git \
    && rm -rf /var/lib/apt/lists/*

# ── Qwen3-TTS: separate venv to avoid dependency conflicts ───
RUN python3 -m venv /app/qwen3tts-venv

RUN /app/qwen3tts-venv/bin/pip install --no-cache-dir \
        torch==2.11.0 torchaudio==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu128 \
    && /app/qwen3tts-venv/bin/pip install --no-cache-dir \
        qwen-tts fastapi uvicorn[standard] soundfile

COPY qwen3tts_server.py /app/qwen3tts_server.py

# ── Orpheus TTS: llama.cpp token generation + SNAC decoder ───
RUN python3 -m venv /app/orpheus-venv

RUN /app/orpheus-venv/bin/pip install --no-cache-dir \
        torch==2.11.0 torchaudio==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu128 \
    && /app/orpheus-venv/bin/pip install --no-cache-dir \
        snac fastapi uvicorn[standard] requests numpy

COPY orpheus_server.py /app/orpheus_server.py
