# Wan2GP — Video Generator for the GPU Poor
# https://github.com/deepbeepmeep/Wan2GP
#
# Toolchain per upstream docs/INSTALLATION.md (RTX 20xx-50xx):
#   Python 3.11 · PyTorch 2.10.0 · CUDA 13 (cu130 wheels) · SageAttention 2.2.0
#
# Base supplies nvcc for the SageAttention build. Any CUDA 13.x works with the
# cu130 wheels — torch only errors on a *major* version mismatch (a minor skew
# just logs a warning), and CUDA minor version compatibility covers the runtime.
# 13.1 is what upstream's install table pairs with the cu130 wheel index.
ARG CUDA_VERSION=13.1.2
FROM docker-proxy.private.krystof.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG WAN2GP_COMMIT=70e194459e8d9dc39a1820a120d13d8e3ba123ef

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 ships Python 3.10, but requirements.txt gates the CUDA-13 build of
# onnxruntime-gpu behind `python_version >= "3.11"` — on 3.10 you get the CUDA-12
# build and ONNX inference breaks. deadsnakes provides 3.11 on jammy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        git wget curl cmake ninja-build build-essential \
        libgl1 libglib2.0-0 libsndfile1 libportaudio2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Jammy's apt ffmpeg is 4.4.2, which predates `-fps_mode` (added in ffmpeg 5.0).
# WanGP's decoder passes that flag unconditionally
# (shared/utils/video_decode.py, _decode_contiguous_video_frames_ffmpeg), so on
# 4.4.2 *any* task with a `video_source` dies with
#   Unrecognized option 'fps_mode'. Error splitting the argument list
# That is every sliding-window continuation — video-extend, and the whole
# boxart roundtrip pipeline. It fails at generation time, not validation, so a
# --dry-run gate reports the queue as perfectly valid first.
#
# Overlay a static build ahead of it. /usr/local/bin precedes /usr/bin in PATH,
# so this shadows the apt binaries for every consumer without removing them —
# the apt package is still wanted for its codec libraries, which torchaudio and
# opencv link against.
#
# Deliberately NOT dropped into /workspace/ffmpeg_bins, the directory WanGP
# checks first (shared/ffmpeg_setup.py:118): an ffmpeg found there is treated as
# a local install and _libs_present() (:139) then demands libavdevice.so*
# beside it. A static binary has no such library, so WanGP concludes the install
# is incomplete and re-downloads a 60 MB bundle on *every* invocation. Resolved
# via PATH instead, it short-circuits to "system-installed" and skips that.
# Setting $FFMPEG_BINARY does not work either — ffmpeg_setup.py:151 overwrites
# it at import.
#
# The URL is a moving "latest release" pointer, so the version is asserted
# rather than assumed: fail the build here rather than ship an image whose
# continuation passes are silently broken.
RUN curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
      | tar xJ -C /tmp \
    && mv /tmp/ffmpeg-*-amd64-static/ffmpeg /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg-*-amd64-static \
    && chmod 0755 /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && /usr/local/bin/ffmpeg -v error -f lavfi -i testsrc=d=0.1 \
         -fps_mode passthrough -f null - \
    && /usr/local/bin/ffprobe -version >/dev/null \
    # and prove PATH resolves to the new one, not the apt 4.4.2 still present
    && test "$(command -v ffmpeg)" = /usr/local/bin/ffmpeg \
    && test "$(command -v ffprobe)" = /usr/local/bin/ffprobe \
    && ffmpeg -version | head -1

# Isolated venv so the system python3.10 (used by apt tooling) stays untouched
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv

RUN pip install --no-cache-dir --upgrade pip wheel packaging ninja

# PyTorch first so this layer survives WAN2GP_COMMIT bumps.
# cu130 wheels bundle their own CUDA runtime; the base image supplies nvcc.
# torch pulls in pytorch-triton, which SageAttention and torch.compile require.
RUN pip install --no-cache-dir \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu130

WORKDIR /workspace

# Clone Wan2GP at pinned commit (update WAN2GP_COMMIT to upgrade)
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /workspace \
    && cd /workspace \
    && git checkout ${WAN2GP_COMMIT} \
    && rm -rf .git

# requirements.txt carries its own --extra-index-url for the CUDA-13
# onnxruntime nightly feed, so no extra flags are needed here.
RUN pip install --no-cache-dir -r requirements.txt

# SageAttention 2 — built from source for the target GPU arch.
#
# CUDA_ARCHITECTURES must be ';'-separated: SageAttention's setup.py splits
# TORCH_CUDA_ARCH_LIST on ','/';' only, so a space-separated "8.9 12.0" is read
# as one token, matches startswith("8.9"), and silently drops sm_120.
#   12.0 = Blackwell  (RTX PRO 5000, RTX 50xx)  -> requires CUDA >= 12.8
#    8.9 = Ada        (RTX 4070 Ti Super, 40xx) -> requires CUDA >= 12.4
#    8.6 = Ampere     (RTX 30xx)
# Only 8.0/8.6/8.9/9.0/12.0 are supported by SageAttention 2; sm_75 and below
# are also dropped by CUDA 13 itself. List only what you deploy on — each arch
# roughly doubles build time.
ARG CUDA_ARCHITECTURES="12.0"
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"

# Build parallelism is capped as a precaution against OOM on a shared machine.
# SageAttention's setup.py feeds MAX_JOBS into *two* levels at once: distutils
# `parallel` across the 3 extension modules, and ninja's job count within each.
# MAX_JOBS=8 was measured at ~9 concurrent nvcc and a ~25GB peak — fine on a
# dedicated builder, tight on a workstation. EXT_PARALLEL=1 serializes the
# modules so MAX_JOBS is the only knob. Budget ~3GB per job; raise for speed.
ARG MAX_JOBS=4
ENV MAX_JOBS="${MAX_JOBS}"
ARG EXT_PARALLEL=1
ENV EXT_PARALLEL="${EXT_PARALLEL}"

# setuptools must be <=75.8.2 for the SageAttention build (upstream instruction)
RUN pip install --no-cache-dir --force-reinstall "setuptools<=75.8.2" \
    && git clone --branch v2.2.0 --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sageattention \
    && cd /tmp/sageattention \
    && pip install --no-cache-dir --no-build-isolation . \
    # `cd /` before the rm is load-bearing: deleting the shell's own working
    # directory leaves every later command with a CWD that no longer exists, and
    # MKL's loader calls getcwd() when resolving the library it dlopen()s. That
    # fails with ENOENT and torch dies with the very misleading
    #   Intel oneMKL FATAL ERROR: Cannot load .../libtorch_cpu.so
    # even though the file is intact and SageAttention built fine.
    && cd / \
    && rm -rf /tmp/sageattention \
    # Fail the build rather than ship an image whose SageAttention has no kernel
    # for the requested arch (setup.py emits an empty gencode list on a typo).
    # _qattn_sm80/_fused are built for every supported arch — the sm80/sm89 in
    # the module names is the kernel source family, not the target GPU: on a
    # 12.0-only build these are compiled with arch=compute_120,code=sm_120.
    && python -c "import sageattention, sageattention._qattn_sm80, sageattention._fused"

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

# Plugins installed at runtime through the WanGP Plugin Manager shell out to
#   pip install -r <plugin>/requirements.txt          (shared/utils/plugins.py)
# as UID 1000. /opt/venv is root-owned and lives in an image layer, so those
# writes would both fail outright and be lost on restart. Redirect them to a
# directory K8s backs with a PVC, and put it on sys.path with a .pth file —
# .pth paths are *appended* by site.addsitedir, so a plugin dependency can
# never shadow torch/numpy/opencv from the venv itself.
#
# PIP_TARGET must stay below every build-time pip install above, or it would
# hijack them too. Note it also makes pip ignore already-installed packages:
# a plugin listing a heavy dep re-downloads it here rather than reusing the
# venv copy. Harmless (the venv copy still wins on import), just disk.
ENV PLUGIN_SITE=/workspace/plugin-site
ENV PIP_TARGET=${PLUGIN_SITE}
ENV PIP_CACHE_DIR=/home/user/.cache/pip
RUN mkdir -p ${PLUGIN_SITE} /home/user/.cache/pip \
    && python -c "import sysconfig, pathlib, os; pathlib.Path(sysconfig.get_paths()['purelib'], 'zz-plugin-site.pth').write_text(os.environ['PLUGIN_SITE'] + chr(10))" \
    && chown -R user:user ${PLUGIN_SITE} /home/user/.cache

EXPOSE 7860

USER user

CMD ["python3", "wgp.py", "--listen"]
