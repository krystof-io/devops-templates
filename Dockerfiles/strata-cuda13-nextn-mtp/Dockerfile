# ─────────────────────────────────────────────────────────────────────
# strata-cuda13-nextn-mtp — Strata (Niko1221/Strata), built FROM SOURCE with one
# local patch: engine support for a qwen4exp GGUF that declares its own
# MTP/nextn prediction block (qwen4exp.nextn_predict_layers > 0).
#
# WHY THIS EXISTS
#   The published image (ghcr.io/niko1221/strata:0.2.3-cuda13, what we run in
#   manual/strata/ today) rejects any qwen4exp file whose block_count includes
#   an MTP/nextn block: the architecture check compares the observed
#   block_count straight against the compiled trunk size (no subtraction for
#   nextn_predict_layers), the native-dense loader has no suffix entries for
#   the three .nextn.* 2-D projections, and iq_pack.py's own expert table
#   emits an expert row for that block that the engine then refuses to load
#   ("a malformed line") because its parser bounds-checks against a
#   trunk-sized n_layers.
#
#   Confirmed hit on a real file this way: orcarouter/Qwen3.8-Flash-Next-
#   Uncensored-GGUF's IQ4_XS-MTP quant (qwen4exp.block_count=49,
#   nextn_predict_layers=1) — crash-loops ghcr.io/niko1221/strata:0.2.3-cuda13
#   in production, exit 1, EngineDied, same "a malformed line: 48 8 8 ..."
#   on every restart.
#
#   This is a KNOWN gap, not a guess: Niko1221/Strata PR #619 (mw00) and its
#   rebase PR #1343 (rluisr) wrote, built and measured the exact 3-file fix
#   against a different model with the identical shape (CYBER-FROST-3.8 /
#   Blackfrost, same block_count/nextn_predict_layers pattern) — real CUDA
#   build, real load, 85.7% MTP draft acceptance over 274 windows. Both were
#   closed unmerged: "Thank you. Model support is outside this speed pass."
#   (Niko1221, PR #1343) — rejected on SCOPE, not correctness.
#
#   The upstream patch (commit 94a3ae2, https://github.com/Niko1221/Strata/
#   commit/94a3ae2.diff) does not `git apply` cleanly against current tags —
#   the surrounding code (Qwen4ExpGuard -> read_geometry/geometry_unsupported/
#   check_architecture split) has been refactored since. patches/
#   0001-nextn-mtp-blocks.patch here is the SAME three changes, hand-re-
#   applied to v0.2.6's current code shape, confirmed to `git apply --check`
#   cleanly against a fresh v0.2.6 clone. See patches/README.md.
#
# WHAT THE PATCH DOES (3 files, 30 insertions, 2 deletions)
#   1. include/strata/artifact/gguf_reader.hpp — read_geometry() now reads
#      qwen4exp.nextn_predict_layers (absent -> 0) and subtracts it from
#      g.n_layers before geometry_unsupported() compares against the
#      compiled trunk size. Mirrors what check_architecture_qwen35moe()
#      already does for the OTHER model family two functions above it.
#   2. src/core/native_dense.cpp — eligible() gains the three 2-D nextn
#      projection suffixes (.nextn.eh_proj/.hc_head_down/.hc_head_up.weight)
#      so they load natively instead of hitting the broken packed fallback.
#      The 1-D nextn norms stay excluded by the existing shape.size()==2 test.
#   3. tools/iq_pack.py — expert_layout() subtracts the declared nextn count
#      from n_layers so native_experts.txt covers the trunk only. Matches
#      ORCA.md's CYBER_FROST.md guidance verbatim: "Check that table has 48
#      rows and no reference to blk.48."
#
#   A file WITHOUT an MTP/nextn block (every model we've run so far: Q2_0,
#   Orca IQ2_M, etc.) reads nextn_predict_layers as absent -> 0 -> every new
#   code path is a no-op. This patch is additive; it does not change how any
#   currently-working model loads.
#
# WHAT THIS DOES NOT DO
#   This does not change tools/mtp_fetch.py, mtp_pack.py or mtp_rt.py (the
#   separately-fetched draft-head flow for models that ship no MTP weights
#   of their own) — IQ4_XS-MTP needs none of that, its MTP head is embedded
#   in the GGUF itself (blk.48.nextn.* tensors), which is exactly the case
#   this patch unblocks.
#
# BUILD (same pattern as llama-swap-unified-cuda13 in this repo):
#   docker build -t docker-private.privpub.krystof.io/krystof-io/strata-cuda13-nextn-mtp:0.2.6 .
#   docker push  docker-private.privpub.krystof.io/krystof-io/strata-cuda13-nextn-mtp:0.2.6
#
#   Narrow CUDA_ARCHITECTURES to our actual cards (sm_120 = RTX PRO 6000 /
#   5000 Blackwell) for a much faster build than upstream's 5-arch default —
#   see ARG below. Compiling 5 archs takes ~25 min per upstream's own
#   Dockerfile comment; sm_120 alone is a fraction of that.
#
#   No GPU needed to build (same as upstream's own Dockerfile): CUDA_ARCH is
#   given explicitly, not read from nvidia-smi.
# ─────────────────────────────────────────────────────────────────────

# ── Version pinning ──────────────────────────────────────────────
# Strata:     v0.2.6, commit 4b9e05a2a806a3fe24f7b2560d98626c37dade35
#             (2026-10-25). Matches the latest upstream release as of this
#             image's creation. The currently-running prod image is
#             ghcr.io/niko1221/strata:0.2.3-cuda13 — this is a version BUMP,
#             not a same-version patch; re-check nothing else moved
#             (docs/MODELS.md's own model list, default flags) if something
#             about the non-MTP models misbehaves after switching to this
#             image. If so, bisect by trying the patch against an UNPATCHED
#             v0.2.6 build first, to separate "0.2.3 -> 0.2.6" drift from
#             "the patch broke something."
# CUDA:       13.0.0, matching upstream Dockerfile's own default
#             (nvidia/cuda:13.0.0-devel-ubuntu24.04) and our driver's ceiling
#             (worker-4 advertised CUDA 13.2 in llama-swap-unified-cuda13's
#             own pin comment; 13.0 is within that ceiling).
# ── Version pinning ──────────────────────────────────────────────
# Strata:     v0.2.6 -> v0.2.8, commit 878ef09cc9092a41a37dac32fd6fe461e47fa326
#             (2026-10-29). Bumped SPECIFICALLY to carry PR #4613 (see
#             patches/0002-unread-call-marker.patch below) — that PR's own
#             base commit (46c152a, 2026-10-28) sits between v0.2.7 and
#             v0.2.8, so v0.2.8 is the FIRST tagged release downstream of
#             it. Confirmed the patch isn't itself IN v0.2.8 yet (PR is
#             still open/unmerged as of this bump) via a direct substring
#             check of serve/frontend.py's unread_info field — absent.
#             70 commits sit between the PR's base and v0.2.8 (mostly
#             SYCL/DeepSeek engine work, unrelated) — `git apply --check`
#             verified clean against a fresh v0.2.8 clone before trusting
#             this pin (see patches/README.md's own verification command,
#             same discipline as the existing nextn-mtp patch). The
#             nextn-mtp patch (0001) was ALSO re-verified against v0.2.8
#             specifically for this bump, not just carried forward — see
#             that patch's own README entry for the result.
ARG STRATA_VERSION=v0.2.8
ARG CUDA_IMAGE=docker-proxy.private.krystof.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG RUNTIME_BASE=docker-proxy.private.krystof.io/ubuntu:24.04

# CUDA_ARCHITECTURES: our cards only (sm_120 = RTX PRO 6000 / 5000 Blackwell).
# Upstream's own default is "75;80;86;89;120" (RTX 20 through 50) — narrowing
# to just our card is the single biggest build-time lever here.
ARG CUDA_ARCHITECTURES="120"
ARG BUILD_VISION=1

# ---- 1. fetch upstream at the pinned tag, apply the local patch
FROM ${CUDA_IMAGE} AS src
ARG STRATA_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${STRATA_VERSION}" https://github.com/Niko1221/Strata.git .

# `git apply --check` FIRST so a patch that no longer applies against a newer
# STRATA_VERSION fails the build LOUDLY instead of silently shipping an
# unpatched (and therefore non-functional, for our purpose) binary — same
# discipline as llama-swap-unified-cuda13's sd.cpp LoKr patch step in this repo.
# 0002 added 2026-10-29: PR #4613 (serve/, no engine code) — makes a
# <tool_call>-shaped block inside a code fence/inline-code COUNTED and
# NAMED in strata_rejected_tool_calls, same as the already-working bare
# and malformed paths. Fixes a real diagnostics gap (#4071): a model
# writing a declared-only or fenced call currently vanishes with zero
# signal on either path — see patches/README.md's own writeup for the
# live GitHub-issue thread that surfaced this (multiple independent
# reporters, including one hitting the identical shape through
# llama-swap, not just Strata) and why #4613 is the right fix (does NOT
# change which blocks stay text — #3075/#1058's rules are explicitly
# preserved — only makes the previously-silent fenced case visible to a
# client via unread_info/strata_rejected_tool_calls). Still unmerged
# upstream as of this bump (open PR, not a maintainer rejection) — same
# "real fix, not yet shipped" situation as 0001.
COPY patches/0001-nextn-mtp-blocks.patch patches/0002-unread-call-marker.patch /patches/
RUN set -eux; \
    for p in /patches/*.patch; do \
        echo "=== applying $(basename "$p") ==="; \
        git apply --check "$p"; \
        git apply "$p"; \
    done; \
    echo "=== patched source ready ==="

# ---- 2. compile the engine and make its component package
#         (identical to upstream's own Dockerfile stage 1, source now patched)
FROM ${CUDA_IMAGE} AS engine-build
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PATH=/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git libgomp1 libatomic1 python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY --from=src /src .
ARG CUDA_ARCHITECTURES
ARG BUILD_VISION
ARG ENGINE_VERSION=
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-build.txt
# llama.cpp at the pinned commit (ggml for the engine, mtmd for pictures), exactly as setup.py fetches it
RUN /venv/bin/python -c "import setup; print(setup.get_llama_cpp())"
RUN set -e; \
    real=""; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}; \
    last=$(echo "$CUDA_ARCHITECTURES" | tr ';,' '  ' | awk '{print $NF}'); \
    sm60=OFF; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do if [ "$a" -lt 75 ]; then sm60=ON; fi; done; \
    /venv/bin/cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_EXPERIMENTAL_SM60=$sm60 \
        -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR=/src/third_party/llama.cpp; \
    /venv/bin/cmake --build build --target strata strata_kq -j "$(nproc)"; \
    if [ "$BUILD_VISION" = "1" ]; then \
      /venv/bin/cmake -G Ninja -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/src/third_party/llama.cpp \
          -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"; \
      /venv/bin/cmake --build build-vision --target strata-vision -j "$(nproc)"; \
    fi
RUN set -e; \
    mkdir -p /stage /engine; \
    cp build/strata /stage/; \
    [ -f build/libstrata_kq.so ] && cp build/libstrata_kq.so /stage/ || true; \
    [ -f build-vision/bin/strata-vision ] && cp build-vision/bin/strata-vision /stage/ || true; \
    ver="${ENGINE_VERSION:-$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)}"; \
    cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1); cudamajor=${cudafull%%.*}; \
    if [ "$cudamajor" = 13 ]; then mindriver=580.65; else mindriver=575.51; fi; \
    vision_archs=""; [ -f /stage/strata-vision ] && vision_archs=", \"vision_archs\": \"$CUDA_ARCHITECTURES\""; \
    echo "{\"version\": \"$ver\", \"source\": \"docker build (nextn-mtp patch)\", \"archs\": \"$CUDA_ARCHITECTURES\", \"cuda\": \"$cudafull\", \"portable\": true${vision_archs}}" > /stage/BUILD.json; \
    python3 packaging/collect_libs.py --binary build/strata --search /usr/local/cuda/lib64 --out /stage --also 'libcublasLt.so*'; \
    archs=""; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do archs="$archs --arch sm_$a"; done; \
    last=$(echo "$CUDA_ARCHITECTURES" | tr ';,' '  ' | awk '{print $NF}'); \
    python3 packaging/package_engine.py --src /stage --backend cuda --version "$ver" --out /engine $archs --arch "compute_$last" \
        --min-driver nvidia=$mindriver --variant cuda$cudamajor --platform linux-x86_64 --build-id "docker-source-${ver}-cuda${cudafull}-nextn-mtp" --include SYSTEM-LIBS.txt --include 'libstrata_kq*' \
        --include strata-vision; \
    for d in /engine/*/; do python3 packaging/package_engine.py --verify "$d"; python3 packaging/binary_deps.py "$d"; done

# ---- 3. the Terminal (the release archive, built here — iq_pack.py's patch lives in this tree)
FROM ${RUNTIME_BASE} AS terminal
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY --from=src /src .
RUN python3 /src/packaging/package_terminal.py --out /out --cache /cache \
 && mkdir -p /opt/strata \
 && tar -xzf /out/strata-terminal-*-linux-x86_64.tar.gz -C /opt/strata \
 && mv /opt/strata/strata-terminal-* /opt/strata/terminal \
 && /opt/strata/terminal/bin/strata --version
# jsonschema, 2026-10-29: serve/structured.py's own json_schema response-format path lazily
# imports this and falls back to "only checked to be one JSON object" (no real schema
# validation) when it's missing — confirmed live on the production pod's own log line:
#   [strata] response_format json_schema: the Python package jsonschema is not installed,
#   so answers are only checked to be one JSON object (python -m pip install "jsonschema>=4.23,<5")
# That log line gives the exact pin; installed straight into the Terminal's bundled
# interpreter (packaging/package_terminal.py's own standalone CPython + locked runtime
# wheels — see that script's own header comment: no internet pip install at USER runtime
# by design, for THEIR release pipeline). This is a one-time BUILD-time install into OUR
# custom image, same category as 0001/0002's source patches above — not fighting their
# hash-locked release mechanism, just adding one optional package their own code already
# knows how to use when present. serve/structured.py imports it lazily (first request that
# actually uses json_schema), so this has zero effect on any model/request that doesn't use
# response_format: json_schema.
RUN /opt/strata/terminal/python/bin/python3 -m pip install --no-cache-dir "jsonschema>=4.23,<5" \
 && /opt/strata/terminal/python/bin/python3 -c "import jsonschema; print('jsonschema', jsonschema.__version__, 'installed OK')"
# llama.cpp at the commit setup.py pins: fetched here, so the first start does not depend on GitHub. Only gguf-py (the model preparation
# tools) and the file setup.py looks for (ggml/CMakeLists.txt) are kept: 170 MB become a few
RUN cd /opt/strata/terminal && STRATA_CONTROL_INSTALL=1 STRATA_INSTALL_DIR=/opt/strata/seed PYTHONNOUSERSITE=1 \
      /opt/strata/terminal/python/bin/python3 -c "import setup; print(setup.get_llama_cpp())" \
 && rm -rf /opt/strata/seed/third_party/*.zip* \
 && cd /opt/strata/seed/third_party/llama.cpp \
 && find . -mindepth 1 -maxdepth 1 ! -name gguf-py ! -name ggml ! -name 'LICENSE*' -exec rm -rf {} + \
 && find ggml -mindepth 1 -depth ! -type d ! -name CMakeLists.txt -delete && find ggml -mindepth 1 -type d -empty -delete \
 && du -sh /opt/strata/seed

# ---- 4. the runtime image (the same layout as upstream's own docker/Dockerfile, FLAVOR=cuda)
FROM ${RUNTIME_BASE}
ARG STRATA_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates libgomp1 libatomic1 libnuma1 zlib1g \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
COPY --from=terminal /opt/strata/terminal /opt/strata/terminal
COPY --from=terminal /opt/strata/seed /opt/strata/seed
COPY --from=engine-build /engine/ /opt/strata/components/engine/
COPY --from=src /src/docker/entrypoint.sh /src/docker/healthcheck.sh /opt/strata/
COPY --from=src /src/docker/strata-wrapper.sh /usr/local/bin/strata
RUN chmod 0755 /opt/strata/entrypoint.sh /opt/strata/healthcheck.sh /usr/local/bin/strata
ENV STRATA_COMPONENTS_DIR=/opt/strata/components STRATA_INSTALL_DIR=/data/install XDG_STATE_HOME=/data/state \
    XDG_CONFIG_HOME=/data/config HOME=/data/home PORT=8080 LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 STRATA_IN_DOCKER=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME ["/data"]
EXPOSE 8080
LABEL org.opencontainers.image.title="Strata (nextn/MTP patch)" com.strata.flavor="cuda" com.strata.build="from-source" \
      org.opencontainers.image.version="${STRATA_VERSION}" com.krystof-io.patch="0001-nextn-mtp-blocks"
HEALTHCHECK --interval=30s --timeout=8s --start-period=2h --retries=3 CMD ["/opt/strata/healthcheck.sh"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"]
CMD ["run"]
