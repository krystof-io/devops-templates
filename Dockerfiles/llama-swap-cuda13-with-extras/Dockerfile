# ── Version pinning ──────────────────────────────────────────────
# CUDA:              https://hub.docker.com/r/nvidia/cuda/tags?name=devel-ubuntu24.04
#                    Driver 595.71.05 on worker-4 advertises CUDA 13.2 as its
#                    ceiling (nvidia-smi), so we pin the 13.2 toolkit — NOT 13.3.
# llama.cpp:         https://github.com/ggml-org/llama.cpp/releases  (bNNNN tags)
# llama-swap:        https://github.com/mostlygeek/llama-swap/releases
# whisper.cpp:       https://github.com/ggml-org/whisper.cpp/releases
# stable-diffusion:  https://github.com/leejet/stable-diffusion.cpp/tags
#                    (no semver — auto-generated master-<N>-<hash> / master-<hash>
#                    tags; the tags API does NOT return them chronologically, and
#                    some recent tags drop the server example. Pinned to the exact
#                    version the working cuda12 image uses so sd-server's CLI stays
#                    byte-for-byte config-compatible — we only recompile on CUDA 13.)
# PyTorch cu130:     https://download.pytorch.org/whl/cu130/torch/
# torchaudio cu130:  https://download.pytorch.org/whl/cu130/torchaudio/
#   Pin both to the SAME version. torchaudio cu130 currently tops out at
#   2.11.0 (torch ships 2.12.0, torchaudio does not), so both pin to 2.11.0.
#   We use --index-url (not --extra-index-url) so pip can ONLY pull the
#   +cu130 wheels — an --extra-index-url lets PyPI's CUDA-13.0 torch win and
#   breaks torchaudio's CUDA version check.
#
# Unlike llama-swap-cuda12-with-extras (which layered onto the prebuilt
# mostlygeek/llama-swap:cuda image — pinned to CUDA 12.8.1 upstream), this
# image builds llama-server, whisper-server and sd-server ALL from source on
# a single CUDA 13.2 base so the whole image is CUDA-13 consistent. It then
# reproduces the upstream llama-swap runtime layout (everything under /app,
# non-root uid 10001, /app on PATH, llama-swap entrypoint).

ARG CUDA_VERSION=13.2.1
ARG UBUNTU_VERSION=24.04
# CUDA compute capabilities to build SASS for (semicolon-separated):
#   89  = Ada Lovelace   — RTX 4070 Ti Super (current worker-4 GPU)
#   120 = Blackwell      — RTX 50-series (incoming GPU)
# Building both makes one fatbinary that runs on either card. Drop to a single
# arch to shrink the image / speed the build once a GPU is retired.
ARG CUDA_ARCH=89;120

# ─────────────────────────────────────────────────────────────────
# Stage 1: Build llama.cpp (llama-server + ggml CUDA backends) from source
# ─────────────────────────────────────────────────────────────────
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama-builder

ARG CUDA_ARCH
ARG LLAMA_CPP_VERSION=b9567

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake git python3 python3-pip \
        libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Match upstream: CUDA 13 host compiler is gcc-14
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

RUN git clone --depth 1 --branch ${LLAMA_CPP_VERSION} \
        https://github.com/ggml-org/llama.cpp.git /app

WORKDIR /app

# Flags mirror ggml-org/llama.cpp .devops/cuda.Dockerfile (server target).
# GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS → CUDA + CPU backends as loadable
# .so files placed next to the binary.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j$(( ($(nproc) + 1) / 2 ))

# Collect the runtime artifacts: shared libs (ggml backends) + llama-server
RUN mkdir -p /artifacts \
    && find build -name "*.so*" -exec cp -P {} /artifacts \; \
    && cp build/bin/llama-server /artifacts/llama-server

# ─────────────────────────────────────────────────────────────────
# Stage 2: Build whisper-server (whisper.cpp) with CUDA support
# ─────────────────────────────────────────────────────────────────
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS whisper-builder

ARG CUDA_ARCH
ARG WHISPER_CPP_VERSION=v1.8.6

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} \
        https://github.com/ggml-org/whisper.cpp.git /whisper.cpp

WORKDIR /whisper.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    && cmake --build build --config Release -t whisper-server -j$(( ($(nproc) + 1) / 2 ))

# ─────────────────────────────────────────────────────────────────
# Stage 3: Build sd-server (stable-diffusion.cpp) with CUDA support
# ─────────────────────────────────────────────────────────────────
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS sd-builder

ARG CUDA_ARCH
# Proven version from the cuda12 image (its server already has the frontend +
# OpenAI/sdapi route rework, so the existing llama-swap config matches it).
ARG SD_CPP_VERSION=master-596-90e87bc

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

RUN git clone --recursive --depth 1 --branch ${SD_CPP_VERSION} \
        https://github.com/leejet/stable-diffusion.cpp.git /stable-diffusion.cpp

WORKDIR /stable-diffusion.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DSD_BUILD_EXAMPLES=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    && cmake --build build --config Release -t sd-server -j$(( ($(nproc) + 1) / 2 ))

# ─────────────────────────────────────────────────────────────────
# Stage 4: Runtime image — CUDA 13.2 runtime + llama-swap + extras
# ─────────────────────────────────────────────────────────────────
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG LLAMA_SWAP_VERSION=223
ARG UID=10001
ARG GID=10001
ARG USER_HOME=/app

# ── Runtime + Python deps for the bundled servers ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 ffmpeg sox git \
        python3 python3-pip python3-venv python3-dev libpython3.12 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── llama.cpp: server binary + ggml CUDA/CPU backend libraries ────
COPY --from=llama-builder /artifacts/ /app/

# ── whisper.cpp + stable-diffusion.cpp servers ────────────────────
COPY --from=whisper-builder /whisper.cpp/build/bin/whisper-server /app/whisper-server
COPY --from=sd-builder /stable-diffusion.cpp/build/bin/sd-server /app/sd-server

# ── llama-swap (Go binary) from GitHub releases ───────────────────
RUN set -eux; \
    case "$(uname -m)" in \
        x86_64)  ARCH=amd64 ;; \
        aarch64) ARCH=arm64 ;; \
        *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl --fail -L -o /tmp/llama-swap.tar.gz \
        "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_${ARCH}.tar.gz"; \
    tar -zxf /tmp/llama-swap.tar.gz -C /app llama-swap; \
    rm /tmp/llama-swap.tar.gz; \
    test -x /app/llama-swap

# ── Qwen3-TTS: separate venv to avoid dependency conflicts ────────
RUN python3 -m venv /app/qwen3tts-venv \
    && /app/qwen3tts-venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchaudio==2.11.0 \
    && /app/qwen3tts-venv/bin/pip install --no-cache-dir \
        qwen-tts fastapi uvicorn[standard] soundfile

COPY qwen3tts_server.py /app/qwen3tts_server.py

# ── Orpheus TTS: llama.cpp token generation + SNAC decoder ────────
RUN python3 -m venv /app/orpheus-venv \
    && /app/orpheus-venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchaudio==2.11.0 \
    && /app/orpheus-venv/bin/pip install --no-cache-dir \
        snac fastapi uvicorn[standard] requests numpy

COPY orpheus_server.py /app/orpheus_server.py

# ── Default config (override by mounting your own /app/config.yaml) ─
COPY config.example.yaml /app/config.yaml

# ── Non-root user, matching upstream llama-swap layout ────────────
ENV HOME=$USER_HOME
RUN groupadd --system --gid $GID app \
    && useradd --system --uid $UID --gid $GID --home $USER_HOME app \
    && chown --recursive $UID:$GID /app

ENV PATH="/app:${PATH}"
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"
WORKDIR /app
USER $UID:$GID

HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]
