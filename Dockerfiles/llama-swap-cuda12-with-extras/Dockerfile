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
    && cmake --build build --config Release -t whisper-server -j$(( $(nproc) / 3 ))

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
    && cmake --build build --config Release -t sd-server -j$(( $(nproc) / 3 ))

# Stage 3: Combine into the llama-swap CUDA image
FROM ghcr-proxy.private.krystof.io/mostlygeek/llama-swap:cuda

COPY --from=whisper-builder /whisper.cpp/build/bin/whisper-server /app/whisper-server
COPY --from=sd-builder /stable-diffusion.cpp/build/bin/sd-server /app/sd-server

# ── F5-TTS: Python + PyTorch + f5-tts package ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev libpython3.12 \
        build-essential ffmpeg sox git \
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

# ── Dia TTS: separate venv (pins torch==2.6.0 + triton) ─────
RUN python3 -m venv /app/dia-venv

RUN /app/dia-venv/bin/pip install --no-cache-dir \
        torch==2.6.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu126 \
    && /app/dia-venv/bin/pip install --no-cache-dir \
        "git+https://github.com/nari-labs/dia.git" fastapi uvicorn[standard] soundfile numpy

COPY dia_server.py /app/dia_server.py

# ── Chatterbox TTS: voice cloning + emotion (exaggeration) ──
RUN python3 -m venv /app/chatterbox-venv

RUN /app/chatterbox-venv/bin/pip install --no-cache-dir \
        torch==2.6.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu126 \
    && /app/chatterbox-venv/bin/pip install --no-cache-dir \
        chatterbox-tts audioseal fastapi uvicorn[standard] soundfile

COPY chatterbox_server.py /app/chatterbox_server.py

# ── CosyVoice 3: voice cloning + emotion instructions ────────
RUN echo 'cosyvoice-rebuild-v8' && git clone --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/cosyvoice-repo

RUN python3 -m venv /app/cosyvoice-venv

RUN /app/cosyvoice-venv/bin/pip install --no-cache-dir \
        torch==2.3.1 torchaudio==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu121

# CosyVoice deps + Matcha-TTS runtime deps (matching official requirements)
RUN /app/cosyvoice-venv/bin/pip install --no-cache-dir \
        conformer==0.3.2 diffusers omegaconf==2.3.0 \
        hydra-core==1.3.2 hydra-colorlog==1.2.0 \
        HyperPyYAML==1.2.3 lightning x-transformers \
        transformers==4.51.3 "numpy<2" \
        openai-whisper soundfile librosa \
        inflect wetext rich numpy scipy einops \
        gdown rootutils Unidecode wget \
        matplotlib pandas seaborn tensorboard \
        pyarrow grpcio networkx onnx pyworld \
        huggingface_hub modelscope protobuf \
        fastapi uvicorn[standard]

# onnxruntime-gpu CUDA 12 build — use --no-deps to prevent it overwriting torch
RUN /app/cosyvoice-venv/bin/pip install --no-cache-dir --no-deps \
        onnxruntime-gpu==1.18.0 \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# pyworld needs pkg_resources, removed from setuptools 71+
RUN /app/cosyvoice-venv/bin/pip install --no-cache-dir 'setuptools<71'

# Force torch back to 2.3.1 — deps like lightning/transformers upgrade it
RUN /app/cosyvoice-venv/bin/pip install --no-cache-dir --force-reinstall \
        torch==2.3.1 torchaudio==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu121

ENV COSYVOICE_LD_PATH=/app/cosyvoice-venv/lib/python3.12/site-packages/torch/lib:/app/cosyvoice-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/cosyvoice-venv/lib/python3.12/site-packages/nvidia/cudnn/lib

RUN echo "=== CosyVoice venv verification ===" \
    && /app/cosyvoice-venv/bin/python -c "import pkg_resources; print('pkg_resources OK')" \
    && /app/cosyvoice-venv/bin/python -c "import pyworld; print('pyworld OK')" \
    && LD_LIBRARY_PATH=${COSYVOICE_LD_PATH} /app/cosyvoice-venv/bin/python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); import torchaudio; print('torchaudio:', torchaudio.__version__)" \
    && LD_LIBRARY_PATH=${COSYVOICE_LD_PATH} /app/cosyvoice-venv/bin/python -c "import onnxruntime; print('onnxruntime:', onnxruntime.__version__, 'providers:', onnxruntime.get_available_providers())" \
    && echo "=== all checks passed ==="

COPY cosyvoice3_server.py /app/cosyvoice3_server.py
