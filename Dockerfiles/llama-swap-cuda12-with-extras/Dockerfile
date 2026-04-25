# Stage 1: Build whisper-server with CUDA support
ARG CUDA_VERSION=12.8.1
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS whisper-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG WHISPER_CPP_VERSION=master

RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} \
        https://github.com/ggml-org/whisper.cpp.git /whisper.cpp

WORKDIR /whisper.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -t whisper-server -j$(( $(nproc) / 2 ))

# Stage 2: Build sd-server (stable-diffusion.cpp) with CUDA support
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS sd-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG SD_CPP_VERSION=master

RUN git clone --recursive --depth 1 --branch ${SD_CPP_VERSION} \
        https://github.com/leejet/stable-diffusion.cpp.git /stable-diffusion.cpp

WORKDIR /stable-diffusion.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DSD_BUILD_EXAMPLES=ON \
    && cmake --build build --config Release -t sd-server -j$(( $(nproc) / 2 ))

# Stage 3: Combine into the llama-swap CUDA image
FROM ghcr-proxy.private.krystof.io/mostlygeek/llama-swap:cuda

COPY --from=whisper-builder /whisper.cpp/build/bin/whisper-server /app/whisper-server
COPY --from=sd-builder /stable-diffusion.cpp/build/bin/sd-server /app/sd-server

# ── F5-TTS: Python + PyTorch + f5-tts package ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libpython3.12 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/f5tts-venv

RUN /app/f5tts-venv/bin/pip install --no-cache-dir \
        torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 \
    && /app/f5tts-venv/bin/pip install --no-cache-dir \
        f5-tts fastapi uvicorn[standard] soundfile

COPY f5tts_server.py /app/f5tts_server.py

# ── Qwen3-TTS: separate venv to avoid dependency conflicts ───
RUN python3 -m venv /app/qwen3tts-venv

RUN /app/qwen3tts-venv/bin/pip install --no-cache-dir \
        torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 \
    && /app/qwen3tts-venv/bin/pip install --no-cache-dir \
        qwen-tts fastapi uvicorn[standard] soundfile

COPY qwen3tts_server.py /app/qwen3tts_server.py
