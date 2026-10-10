# ─────────────────────────────────────────────────────────────────────
# llama-swap-unified-cuda13 — TEST image, parallel to llama-swap-cuda13-with-extras
#
# WHY THIS EXISTS
#   Upstream llama-swap ships a "unified" container build (docker/unified/) that
#   bundles llama.cpp, whisper.cpp, stable-diffusion.cpp, ik_llama.cpp, audio.cpp
#   and a vllm-wrapper, with install scripts upstream maintains. The PUBLISHED
#   image (ghcr.io/mostlygeek/llama-swap:unified-cuda) is NOT usable here:
#
#     • It compiles SASS for "60;61;75;86;89" only. Our cards are RTX PRO 6000 and
#       RTX PRO 5000 Blackwell = sm_120. Upstream's own README says Blackwell
#       "run[s] by JIT-compiling the nearest lower PTX, which costs time on first
#       load" and misses arch-specific kernels.
#     • It is CUDA 12.9.1, not 13.x.
#     • It defaults to RUN_UID=0 (root). Our /models PVC is owned by 10001 and is
#       SHARED with the other llama-swap instance — root-written files would break it.
#
#   So we build upstream's scripts OURSELVES with CUDA_ARCH=120, a CUDA 13.2 base,
#   uid 10001, and our pinned component versions. What we gain over
#   llama-swap-cuda13-with-extras: ik-llama-server, audiocpp_server, vllm-wrapper,
#   llama-bench/llama-cli/llama-tts, and upstream-maintained build logic instead of
#   ~200 lines of hand-rolled cmake.
#
# DROP-IN COMPATIBILITY (deliberate)
#   Upstream installs binaries to /usr/local/bin; the old image used /app. Every
#   `cmd:` in our llama-swap configmaps hardcodes /app/... (30x llama-server,
#   4x sd-server, 2x whisper-server). To make this image A/B testable by changing
#   ONLY the image tag, /app/<bin> symlinks are created for all three. Remove the
#   symlinks once the configs move to a ${bindir} macro.
#
# BUILD:
#   docker build -t \
#     docker-private.privpub.krystof.io/krystof-io/llama-swap-unified-cuda13:latest .
#
#   This file uses --mount=type=cache, which needs BuildKit. No DOCKER_BUILDKIT=1
#   needed: BuildKit has been the default builder since Docker 23.0 (we're on 29.7).
#   Only relevant if something ever builds this on a pre-23 daemon.
#
#   Building sm_120 alone is FASTER than upstream's 5-arch default, but this still
#   compiles five CUDA projects. Expect a long first build; the ccache and per-project
#   build cache mounts make rebuilds cheap.
# ─────────────────────────────────────────────────────────────────────

# ── Version pinning ──────────────────────────────────────────────
# CUDA:          Driver 595.71.05 on worker-4 advertises CUDA 13.2 as its ceiling,
#                so pin the 13.2 toolkit — NOT 13.3. (Same reasoning as the old image.)
#                NOTE: sm_120 needs CUDA >= 12.8, so 12.9 would also work; we stay on
#                13.2 to keep parity with the image this replaces.
# llama-swap:    https://github.com/mostlygeek/llama-swap/releases
#                This tag drives BOTH the llama-swap binary AND the install scripts,
#                so the build logic and the runtime always agree.
#                Old image ran v235; v251 is current. This is a version BUMP — if the
#                test misbehaves, drop to 235 before blaming the unified build.
# EVERY component is bumped to current as of 2026-08-23. That is the point of this
# image: it is a test vehicle, so it moves all axes at once and gets validated as a
# unit. The production image (llama-swap-cuda13-with-extras) keeps the old pins until
# this one passes. If it fails, bisect by reverting ARGs one at a time — the old
# values are recorded below.
#
# llama.cpp:     b10434 -> b10595 -> b11539. The ORIGINAL pin was not arbitrary:
#                `--reasoning-effort` does not exist before b10434 (on b10067
#                llama-server exited in ~800ms with "upstream command exited
#                prematurely"). b11539 is ~950 builds past that floor — re-check
#                `--reasoning-effort`/`--spec-type draft-mtp`/chat-template flags
#                on the two 27B chat entries before trusting this blindly. FIRST
#                THING TO CHECK on any weirdness:
#                  /usr/local/bin/llama-server --help | grep reasoning
# whisper.cpp:   v1.8.6 -> v1.9.3 -> v1.9.5. Patch-version bump; watch for CLI
#                flag changes in the two whisper-server entries.
# sd.cpp:        90e87bc (2026-05-06) -> 97d2990 (2026-08-19) -> f89d9b1
#                (2026-10-09). Bumped specifically to pick up a LoKr LoRA fix —
#                see patches/sdcpp-lokr-multidim-batch.patch below; the fix was
#                NOT available at 97d2990 and still is not merged upstream, so
#                a local patch rides on top of this pin too.
#                Pinned by FULL COMMIT SHA, not a tag: sd.cpp's auto-generated
#                master-<N>-<hash> / master-<hash> tags are not returned
#                chronologically by the API and are genuinely hard to order. A full
#                40-char SHA is unambiguous (git fetch --depth=1 needs the full SHA;
#                an abbreviated one fails with "couldn't find remote ref").
#                RE-VERIFY the 4 sd-server entries' CLI flags after this bump.
# ik_llama.cpp:  no release tags at all upstream, so `main` was the only option.
#                Re-pinned to the SHA main resolved to on 2026-10-09.
# audio.cpp:     re-pinned to the SHA main resolved to on 2026-10-09. Upstream
#                independently fixed the VoxCPM2 streaming-click bug this image
#                used to carry a local patch for (commits a045e690a6 / 6c70f32d0d,
#                2026-09-20 — same technique, a stream_left_context decode
#                window defaulting to 3 patches) — the local
#                voxcpm2-streaming-seam.patch is REMOVED as of this bump since
#                it no longer applies cleanly and is no longer needed.
ARG CUDA_VERSION=13.2.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_ARCH=120
ARG LLAMA_SWAP_VERSION=262
ARG LLAMA_CPP_VERSION=b11539
ARG WHISPER_CPP_VERSION=v1.9.5
ARG SD_CPP_VERSION=f89d9b13d730eabeede7314ce49dacd18d3c90c2
ARG IK_LLAMA_VERSION=5194a9e1451b885e24dd5a065f650864e8807c59
ARG AUDIO_CPP_VERSION=715108a8efde4bb9831dd19771dc3a5efbbdc953

# ── Python pins for the TTS venvs ────────────────────────────────
# The old image pinned torch/torchaudio but left EVERYTHING else floating
# (`pip install qwen-tts fastapi uvicorn[standard] soundfile`), so two builds of
# the same Dockerfile could differ. These are pinned to what PyPI served on
# 2026-08-23 — i.e. freezing what an unpinned build would have installed anyway,
# NOT a version change.
#
# torch/torchaudio stay at 2.11.0. RE-VERIFIED 2026-08-23: torch cu130 now ships
# 2.12.0/2.12.1/2.13.0, but torchaudio cu130 STILL tops out at 2.11.0. They must
# match, so 2.11.0 remains the ceiling. Recheck before bumping either.
# snac / requests / numpy are gone along with the Orpheus venv.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG QWEN_TTS_VERSION=0.1.1
ARG FASTAPI_VERSION=0.141.1
ARG UVICORN_VERSION=0.52.4
ARG SOUNDFILE_VERSION=0.14.0

# ─────────────────────────────────────────────────────────────────
# Stage 1: builder base — toolchain + upstream's install scripts
# ─────────────────────────────────────────────────────────────────
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder-base

ARG CUDA_ARCH
ARG LLAMA_SWAP_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# ccache is NOT optional: upstream's scripts hardcode
# -DCMAKE_C_COMPILER_LAUNCHER=ccache, so cmake fails outright without it.
# The libav*-dev packages are what whisper.cpp needs for WHISPER_FFMPEG=yes.
# libmp3lame-dev is a NEW requirement as of this bump's AUDIO_CPP_VERSION pin:
# install-audio.sh now passes -DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;
# mp3_encode, and the mp3_encode frontend's CMake find_package fails outright
# (lame/lame.h + libmp3lame) without it — confirmed this is upstream's own
# install script requiring it, not something papered over before.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake git python3 python3-pip \
        libssl-dev libgomp1 curl ca-certificates ccache make wget \
        pkg-config libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
        libmp3lame-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13's supported host compiler is gcc-14; Ubuntu 24.04 defaults to gcc-13.
# Upstream gets away without this because it builds on CUDA 12.9.
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

# Consumed by every install-*.sh via -DCMAKE_CUDA_ARCHITECTURES.
# THIS IS THE WHOLE POINT OF THIS IMAGE — 120 = Blackwell native SASS.
ENV CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=2G
ENV PATH="/usr/lib/ccache:${PATH}"

# Fetch upstream's build scripts at the SAME tag as the llama-swap binary.
# Sparse checkout so we pull only docker/unified, not the whole Go project.
RUN set -eux; \
    mkdir -p /build; \
    git clone --depth 1 --filter=blob:none --sparse \
        --branch "v${LLAMA_SWAP_VERSION}" \
        https://github.com/mostlygeek/llama-swap.git /src/llama-swap-repo; \
    git -C /src/llama-swap-repo sparse-checkout set docker/unified; \
    cp /src/llama-swap-repo/docker/unified/install-*.sh /build/; \
    cp /src/llama-swap-repo/docker/unified/lib-release.sh /build/; \
    cp /src/llama-swap-repo/docker/unified/audiocpp-server.example.json /build/; \
    chmod +x /build/install-*.sh /build/lib-release.sh; \
    ls -la /build/

WORKDIR /build

# ─────────────────────────────────────────────────────────────────
# Stage 2: whisper.cpp  (fastest build first, per upstream ordering)
# ─────────────────────────────────────────────────────────────────
FROM builder-base AS whisper-build
ARG WHISPER_CPP_VERSION
RUN --mount=type=cache,id=ccache-cuda13,target=/ccache \
    --mount=type=cache,id=whisper-cuda13,target=/src/whisper.cpp/build \
    BACKEND=cuda WHISPER_FFMPEG=yes bash /build/install-whisper.sh "${WHISPER_CPP_VERSION}"

# ─────────────────────────────────────────────────────────────────
# Stage 3: stable-diffusion.cpp  (sd-server + sd-cli)
# ─────────────────────────────────────────────────────────────────
FROM builder-base AS sd-build
ARG SD_CPP_VERSION
RUN --mount=type=cache,id=ccache-cuda13,target=/ccache \
    --mount=type=cache,id=sd-cuda13,target=/src/stable-diffusion.cpp/build \
    BACKEND=cuda bash /build/install-sd.sh "${SD_CPP_VERSION}"

# ── Local patch: LoKr multi-dim-batch fix (resurrected #1980) ────
# See patches/README.md for the full writeup. Same apply-after-build pattern
# as the audio-build stage below: install-sd.sh's own `git fetch --depth=1
# origin <sha>; git checkout FETCH_HEAD` would fight a working tree patched
# beforehand, and patching afterwards reuses the cmake cache + ccache the
# first build just populated — this recompiles the one changed translation
# unit and relinks rather than rebuilding the whole CUDA project. Both cache
# mounts must be repeated here or the build directory isn't visible.
#
# `git apply --check` runs first so a patch that no longer applies FAILS THE
# BUILD LOUDLY rather than silently shipping an unpatched binary — exactly
# the same reasoning as the audio-build patch step.
COPY patches/sdcpp-lokr-multidim-batch.patch /patches/
RUN --mount=type=cache,id=ccache-cuda13,target=/ccache \
    --mount=type=cache,id=sd-cuda13,target=/src/stable-diffusion.cpp/build \
    set -eux; \
    cd /src/stable-diffusion.cpp; \
    for p in /patches/*.patch; do \
        echo "=== applying $(basename "$p") ==="; \
        git apply --check "$p"; \
        git apply "$p"; \
    done; \
    cmake --build build --config Release -j"$(nproc)" --target sd-server sd-cli; \
    cp build/bin/sd-server /install/bin/; \
    cp build/bin/sd-cli /install/bin/; \
    echo "=== patched sd-server/sd-cli installed ==="

# ─────────────────────────────────────────────────────────────────
# Stage 4: audio.cpp  (audiocpp_server — NEW, not in the old image)
# ─────────────────────────────────────────────────────────────────
# Built as a "deployment build": the model_specs/*.json catalog is compiled into
# the binary, because a container of bare binaries has no model_specs/ dir to find.
# The catalog is also installed to /usr/local/share/audiocpp for --model-spec-override.
FROM builder-base AS audio-build
ARG AUDIO_CPP_VERSION
RUN --mount=type=cache,id=ccache-cuda13,target=/ccache \
    --mount=type=cache,id=audio-cuda13,target=/src/audio.cpp/build \
    BACKEND=cuda bash /build/install-audio.sh "${AUDIO_CPP_VERSION}"

# ── voxcpm2-streaming-seam.patch REMOVED as of AUDIO_CPP_VERSION
# 715108a8efde4bb9831dd19771dc3a5efbbdc953 ──────────────────────
# Upstream fixed the same bug independently (commits a045e690a6 / 6c70f32d0d,
# 2026-09-20), using the same technique our local patch used: decode a window
# of preceding patches as left context and trim them off again
# (stream_left_context, default 3 — matching our kStreamingDecodeContextPatches).
# Confirmed the old patch no longer applies cleanly against this pin
# (`git apply --check` fails on session.cpp), which is exactly the expected
# signal per patches/README.md to check whether upstream fixed it first — and
# it had. No patch step needed for audio.cpp any more.

# ─────────────────────────────────────────────────────────────────
# Stage 5: ik_llama.cpp  (ik-llama-server — NEW)
# ─────────────────────────────────────────────────────────────────
# Alternative llama.cpp fork with the IQK quant family. The reason to care here is
# qwen3-coder-next:80b-a3b — a big MoE is exactly where IQK quants are meant to win.
FROM builder-base AS ik-llama-build
ARG IK_LLAMA_VERSION
RUN --mount=type=cache,id=ccache-cuda13,target=/ccache \
    --mount=type=cache,id=ik-llama-cuda13,target=/src/ik_llama.cpp/build \
    bash /build/install-ik-llama.sh "${IK_LLAMA_VERSION}"

# ─────────────────────────────────────────────────────────────────
# Stage 6: llama.cpp  (slowest build last)
# ─────────────────────────────────────────────────────────────────
# NOTE a real difference from the old image: upstream builds with
# -DBUILD_SHARED_LIBS=OFF, so llama-server is STATIC. The old image used
# GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS, shipping ggml backends as loadable
# .so files next to the binary. Static means no LD_LIBRARY_PATH juggling, but
# also no runtime CPU-variant selection.
FROM builder-base AS llama-build
ARG LLAMA_CPP_VERSION
RUN --mount=type=cache,id=ccache-cuda13,target=/ccache \
    --mount=type=cache,id=llama-cuda13,target=/src/llama.cpp/build \
    BACKEND=cuda bash /build/install-llama.sh "${LLAMA_CPP_VERSION}"

# ─────────────────────────────────────────────────────────────────
# Stage 7: llama-swap binary + vllm-wrapper
# ─────────────────────────────────────────────────────────────────
FROM builder-base AS llama-swap-build
ARG LLAMA_SWAP_VERSION
RUN bash /build/install-llama-swap.sh "${LLAMA_SWAP_VERSION}"

# vllm-wrapper is not in the release archives, so upstream compiles it from the
# same revision as the binary. Needs Go, hence a separate base.
# go1.26 was sufficient at the original pin; v262's go.mod now requires >=1.27.1
# (go: go.mod requires go >= 1.27.1 (running go 1.26.9; GOTOOLCHAIN=local)) — bumped
# to the exact current patch release rather than a floating minor tag.
FROM docker-proxy.private.krystof.io/golang:1.27.2-bookworm AS vllm-wrapper-build
ARG LLAMA_SWAP_VERSION
COPY --from=builder-base /build/install-vllm-wrapper.sh /build/lib-release.sh /build/
RUN --mount=type=cache,id=go-build,target=/root/.cache/go-build \
    --mount=type=cache,id=go-mod,target=/go/pkg/mod \
    bash /build/install-vllm-wrapper.sh "${LLAMA_SWAP_VERSION}"

# ─────────────────────────────────────────────────────────────────
# Stage 8: runtime
# ─────────────────────────────────────────────────────────────────
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG UID=10001
ARG GID=10001
ARG USER_HOME=/app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 ffmpeg sox git \
        python3 python3-pip python3-venv python3-dev libpython3.12 \
        python3-numpy python3-sentencepiece \
        libavcodec60 libavformat60 libavutil58 libswresample4 \
        libmp3lame0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# CUDA stub libcuda, needed because upstream links llama-server against it.
# Kept in the stubs/ subdirectory ONLY — it must never land on the default
# library search path, or it shadows the real driver injected by the NVIDIA
# container runtime and CUDA init fails.
COPY --from=builder-base /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
COPY --from=builder-base /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# ── Binaries (upstream layout: /usr/local/bin) ────────────────────
COPY --from=llama-build      /install/bin/llama-server      /usr/local/bin/
COPY --from=llama-build      /install/bin/llama-cli         /usr/local/bin/
COPY --from=llama-build      /install/bin/llama-tts         /usr/local/bin/
COPY --from=llama-build      /install/bin/llama-bench       /usr/local/bin/
COPY --from=whisper-build    /install/bin/whisper-server    /usr/local/bin/
COPY --from=whisper-build    /install/bin/whisper-cli       /usr/local/bin/
COPY --from=whisper-build    /install/lib/                  /usr/local/lib/
COPY --from=sd-build         /install/bin/sd-server         /usr/local/bin/
COPY --from=sd-build         /install/bin/sd-cli            /usr/local/bin/
COPY --from=sd-build         /install/lib/                  /usr/local/lib/
COPY --from=audio-build      /install/bin/audiocpp_server   /usr/local/bin/
COPY --from=audio-build      /install/bin/audiocpp_cli      /usr/local/bin/
COPY --from=audio-build      /install/share/audiocpp/       /usr/local/share/audiocpp/
COPY --from=ik-llama-build   /install/bin/ik-llama-server   /usr/local/bin/
COPY --from=llama-swap-build /install/bin/llama-swap        /usr/local/bin/
COPY --from=vllm-wrapper-build /install/bin/vllm-wrapper    /usr/local/bin/
COPY --from=builder-base     /build/audiocpp-server.example.json /etc/llama-swap/audiocpp-server.example.json

RUN ldconfig

# ── /app compatibility symlinks — MIGRATION AID, REMOVE LATER ─────
# Our configmaps hardcode these three paths. The symlinks make this image a
# drop-in for llama-swap-cuda13-with-extras so the two can be A/B tested by
# changing only the image tag. Delete once the configs use a ${bindir} macro.
#   30x /app/llama-server   4x /app/sd-server   2x /app/whisper-server
RUN mkdir -p /app \
    && ln -s /usr/local/bin/llama-server   /app/llama-server \
    && ln -s /usr/local/bin/sd-server      /app/sd-server \
    && ln -s /usr/local/bin/whisper-server /app/whisper-server \
    && ln -s /usr/local/bin/sd-cli         /app/sd

# ── Qwen3-TTS venv ────────────────────────────────────────────────
# Carried over unchanged: `qwen3-tts:1.7b-clone` is a member of the
# broadcaster-pipeline group, so dropping it would break broadcaster.
# cu130 wheels match the CUDA 13 base. Pin torch and torchaudio to the SAME
# version — torchaudio cu130 tops out at 2.11.0 while torch ships 2.12.0.
# --index-url (not --extra-index-url) so pip can ONLY see the +cu130 wheels;
# an extra-index lets PyPI's CUDA-13.0 torch win and breaks torchaudio's check.
# INCLUDE_QWEN3TTS=false builds a slim image with NO Python ML stack at all —
# no torch, no torchaudio, ~4 GB smaller, and the torchaudio-cu130-caps-at-2.11.0
# constraint disappears from this file entirely. Do that only once audio.cpp is
# proven to cover the clone case (see below).
#
# Kept ON by default because the live entry is:
#   qwen3-tts:1.7b-clone -> qwen3tts_server.py --mode clone --voices-dir /models/tts-voices
# i.e. ZERO-SHOT VOICE CLONING from reference audio (needs ref_text), using
# Qwen/Qwen3-TTS-12Hz-1.7B-Base. It is a member of the broadcaster-pipeline group,
# so broadcaster prod AND dev depend on it.
#
# audio.cpp lists a qwen3_tts family, but its example config exposes preset voices
# (default_voice_preset.voice_id), not zero-shot cloning — parity is UNVERIFIED.
# The model on the PVC is HF-format (/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base)
# plus /models/tts-voices; audio.cpp is ggml and wants its own package layout, so
# migrating is a model CONVERSION, not a config edit. Keep both present to A/B it.
ARG INCLUDE_QWEN3TTS=true
ARG TORCH_VERSION
ARG TORCHAUDIO_VERSION
ARG QWEN_TTS_VERSION
ARG FASTAPI_VERSION
ARG UVICORN_VERSION
ARG SOUNDFILE_VERSION
RUN if [ "${INCLUDE_QWEN3TTS}" = "true" ]; then \
        python3 -m venv /app/qwen3tts-venv \
        && /app/qwen3tts-venv/bin/pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu130 \
            torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        && /app/qwen3tts-venv/bin/pip install --no-cache-dir \
            qwen-tts==${QWEN_TTS_VERSION} \
            fastapi==${FASTAPI_VERSION} \
            "uvicorn[standard]==${UVICORN_VERSION}" \
            soundfile==${SOUNDFILE_VERSION}; \
    else \
        echo "INCLUDE_QWEN3TTS=false — skipping Python TTS venv (audio.cpp only)"; \
    fi

COPY qwen3tts_server.py /app/qwen3tts_server.py

# ── Orpheus TTS: DROPPED ──────────────────────────────────────────
# The old image built a second full torch+torchaudio venv for Orpheus. A grep of
# the whole cluster repo (excluding manual/archive) finds ZERO references to
# /app/orpheus_server.py or /app/orpheus-venv — no model entry, no group member,
# nothing. It was pure dead weight, costing ~4 GB.
#
# Deleted rather than made optional: there is no config to un-break. If Orpheus is
# ever wanted again, the server script is still in git history alongside the old
# llama-swap-cuda13-with-extras image.

# ── Default config (overridden by the mounted ConfigMap) ──────────
COPY config.example.yaml /app/config.yaml

# ── Non-root user, uid/gid 10001 ──────────────────────────────────
# MUST match the old image and the fsGroup on the shared llama-swap-models PVC.
# Upstream's RUN_UID defaults to 0; root here would write root-owned files into
# a volume the OTHER llama-swap instance reads as 10001.
ENV HOME=$USER_HOME
RUN groupadd --system --gid $GID app \
    && useradd --system --uid $UID --gid $GID --home $USER_HOME app \
    && mkdir -p /etc/llama-swap/config \
    && chown --recursive $UID:$GID /app /etc/llama-swap

ENV PATH="/usr/local/bin:/app:${PATH}"
WORKDIR /app
USER $UID:$GID

HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1

# The config path lives in CMD, NOT in ENTRYPOINT. Kubernetes `args:` replaces
# CMD but never ENTRYPOINT, so baking `-config /app/config.yaml` into ENTRYPOINT
# would leave it in effect alongside whatever the pod passes. That is harmless
# when the pod also passes `-config` (Go's flag package takes the last
# occurrence) but BREAKS with `-config-dir`, which is a different flag: both
# sources load, get merged, and the stock config.example.yaml collides with the
# real config on healthCheckTimeout.
# Splitting them this way lets `args: ["-config-dir", "/config"]` cleanly replace
# the default, with no `command:` override needed in the manifest.
ENTRYPOINT [ "/usr/local/bin/llama-swap" ]
CMD [ "-config", "/app/config.yaml" ]
