# Stage 1: Build whisper-server with CUDA support
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

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
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -t whisper-server -j$(nproc)

# Stage 2: Add whisper-server to the llama-swap CUDA image
FROM ghcr.io/mostlygeek/llama-swap:cuda

COPY --from=builder /whisper.cpp/build/bin/whisper-server /app/whisper-server
