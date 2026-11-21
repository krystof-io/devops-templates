# AVTR-1 — Avaturn open-weights real-time talking-head model (offline workbench)
# https://github.com/avaturn-live/avtr-1
#
# Everything that can be pinned is pinned, because the whole point of this image
# is a supply-chain-clean build (see kio-public-cluster manual/avtr1/README.md
# for the dependency audit this was written against):
#   - base image by digest
#   - pixi by release + sha256
#   - upstream repo by commit
#   - both Hugging Face revisions (upstream floats on "main") by commit SHA
#   - every conda/PyPI package by the sha256 in upstream's committed pixi.lock
#     (`pixi install --frozen`)
#   - the one thing the lock does NOT hash — the real TensorRT libraries, see below
#
# No weights and no tokens are baked in. Artifacts live on the PVC at /data.
#
# Build (hand-built, like the other GPU images):
#   docker build -t docker-private.privpub.krystof.io/krystof-io/avtr1:eb1e5a8 .
#   docker push docker-private.privpub.krystof.io/krystof-io/avtr1:eb1e5a8

# The lock is a CUDA 12 stack (torch 2.7.1+cu128, tensorrt_cu12 10.11, nvcc 12.9).
# torch's wheels bundle their own CUDA 12.8 runtime + cuDNN 9.7; the base supplies
# system CUDA/cuDNN 12.9 as the fallback onnxruntime-gpu looks for. 12.9.1 is the
# newest 12.x tag NVIDIA publishes (last rebuilt 2025-07-25).
FROM docker-proxy.private.krystof.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:d02c4310b6d57ca0b16cd80298bdb33a74187baafe2eccd8a6a16180ddc90802

ARG AVTR1_COMMIT=eb1e5a89ed7bec3debe712b090efd676b3a420bd
# Hugging Face commits resolved 2026-11-20. Changing either changes the storage
# directory name ($AVTR1_LOCAL_STORAGE/<AVTR1_HF_SHA>/) and invalidates engines.
ARG AVTR1_HF_SHA=4c9bd5550f2617d0409ac602f75378502f675731
ARG DITTO_HF_SHA=e4a2f60328ee7c32af585ac4b3cce299e4c8e254

# v0.83.0 (2026-10-22): deliberately not the newest release, so it has had a few
# weeks in the wild. sha256 is of pixi-x86_64-unknown-linux-musl.tar.gz.
ARG PIXI_VERSION=v0.83.0
ARG PIXI_SHA256=064f371e692aaa3400ac581c89da2843c91437f20c2d9155395a5d0037f33551

# The real TensorRT libraries wheel. PyPI only carries a 709-byte stub sdist for
# tensorrt_cu12_libs; at install its build backend (wheel-stub) downloads this
# wheel from pypi.nvidia.com, so pixi.lock hashes the stub and not the ~GB of
# native code that actually lands in the env. This sha256 is the one NVIDIA's
# index publishes for the file; the build fails below if what got installed
# differs from it.
ARG TRT_LIBS_WHEEL=tensorrt_cu12_libs-10.11.0.33-py2.py3-none-manylinux_2_28_x86_64.whl
ARG TRT_LIBS_SHA256=81ace8d3284fdbef0804c444a4d7555343ee079370e79c93cb328c7d9b08f968

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg, libGL, nvcc and the compiler toolchain all come from the pixi env.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/pixi.tar.gz \
      "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
    && echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi \
    && chmod 0755 /usr/local/bin/pixi \
    && rm /tmp/pixi.tar.gz \
    && pixi --version

# Independently fetched, hash-checked copy of the TensorRT libs, kept only to
# compare against what the install step produces.
RUN curl -fsSL -o "/tmp/${TRT_LIBS_WHEEL}" "https://pypi.nvidia.com/tensorrt-cu12-libs/${TRT_LIBS_WHEEL}" \
    && echo "${TRT_LIBS_SHA256}  /tmp/${TRT_LIBS_WHEEL}" | sha256sum -c - \
    && mkdir -p /opt/trt-verified \
    && unzip -q "/tmp/${TRT_LIBS_WHEEL}" -d /opt/trt-verified \
    && rm "/tmp/${TRT_LIBS_WHEEL}"

# ubuntu:24.04 already ships uid/gid 1000 as "ubuntu"; the manifests run as 1000.
RUN mkdir -p /opt/avtr-1 /data && chown 1000:1000 /opt/avtr-1 /data
USER 1000:1000
ENV HOME=/home/ubuntu
WORKDIR /opt/avtr-1

RUN git clone https://github.com/avaturn-live/avtr-1.git . \
    && git checkout --detach "${AVTR1_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${AVTR1_COMMIT}"

# Pin both Hugging Face sources. Upstream ships `HF_REVISION = "main"` and
# `revision="main"` for ditto; each sed must change exactly one line or the
# build stops, so an upstream refactor can't silently leave one floating.
RUN set -eu; f=src/avtr1_renderer/artifact_configs/v1.py; \
    test "$(grep -c '^HF_REVISION = "main"$' "$f")" = 1; \
    test "$(grep -c 'revision="main")$' "$f")" = 1; \
    sed -i "s|^HF_REVISION = \"main\"\$|HF_REVISION = \"${AVTR1_HF_SHA}\"|" "$f"; \
    sed -i "s|revision=\"main\")\$|revision=\"${DITTO_HF_SHA}\")|" "$f"; \
    ! grep -n '"main"' "$f"; \
    grep -n "${AVTR1_HF_SHA}\|${DITTO_HF_SHA}" "$f"

# No GPU at build time: tell pixi the CUDA virtual package is satisfied.
ENV CONDA_OVERRIDE_CUDA=12.8

# The three tensorrt sdists build in isolation and pull their build tools
# (wheel-stub, setuptools, wheel) unpinned from PyPI. Constrain the one that
# does the downloading. NOTE: whether pixi's embedded uv honours this variable
# is unverified — the version actually used is printed below, check the log.
RUN printf 'wheel-stub==0.5.0\n' > /tmp/build-constraints.txt
ENV UV_BUILD_CONSTRAINT=/tmp/build-constraints.txt

# `renderer` only (dev + build features). The `streamer` env — aiortc and the
# cloud conversation engines — is never installed.
RUN pixi install --frozen -e renderer \
    && echo "--- build tools found in the pixi/uv cache:" \
    && (find "$HOME/.cache" -iname 'wheel_stub*' -o -iname 'setuptools-*' -o -iname 'wheel-0*' 2>/dev/null | sed 's|.*/||' | sort -u || true)

# TensorRT libs: what wheel-stub installed must be byte-identical to the
# hash-checked wheel fetched above.
RUN set -eu; sp=.pixi/envs/renderer/lib/python3.12/site-packages; \
    test -d "$sp/tensorrt_libs"; \
    diff -r -x __pycache__ /opt/trt-verified/tensorrt_libs "$sp/tensorrt_libs"; \
    echo "tensorrt_libs matches ${TRT_LIBS_SHA256}"

# Record exactly what is in the env, for diffing against the audited lock.
RUN pixi list --frozen -e renderer > /opt/avtr-1/installed-packages.txt \
    && pixi clean cache --yes || true

# Runtime defaults: artifacts on the PVC, no Hub access, no STUN, never re-solve.
# Always run tasks as `pixi run -e renderer <task>`: the lock also carries a
# smaller `default` env that was not installed.
ENV AVTR1_LOCAL_STORAGE=/data/artifacts \
    HF_HOME=/data/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STUN_URL="" \
    PIXI_FROZEN=true

CMD ["sleep", "infinity"]
