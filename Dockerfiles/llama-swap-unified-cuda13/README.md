# llama-swap-unified-cuda13

**Status: test image.** Parallel to `llama-swap-cuda13-with-extras`, which stays the
production image until this one is validated.

Built from upstream llama-swap's `docker/unified` install scripts instead of
hand-rolled cmake, but with our CUDA version, our GPU arch, our UID, and our
component pins.

## Why not just use `ghcr.io/mostlygeek/llama-swap:unified-cuda`

Three blockers, in order of severity:

| | published `unified-cuda` | this image |
|---|---|---|
| CUDA arch (SASS) | `60;61;75;86;89` — **no Blackwell** | `120` native |
| CUDA toolkit | 12.9.1 | 13.2.1 |
| Runtime UID | 0 (root) | 10001 |

Our cards are RTX PRO 6000 and RTX PRO 5000 Blackwell, both **sm_120**. Upstream's
own README says Blackwell "run[s] by JIT-compiling the nearest lower PTX, which
costs time on first load" and misses arch-specific kernels. The root default is
worse than it sounds: the `llama-swap-models` PVC is owned by 10001 and is
**shared between both llama-swap instances**, so root-written files break the other one.

Everything else about the unified build is worth having, hence this file.

## What's new vs `llama-swap-cuda13-with-extras`

Gained:

- **`ik-llama-server`** — ik_llama.cpp, the IQK-quant fork. Most interesting for
  `qwen3-coder-next:80b-a3b`, since big MoE is where IQK quants are supposed to win.
- **`audiocpp_server`** — audio.cpp TTS/audio. Potential replacement for *both*
  Python TTS venvs (~4 GB of torch each).
- **`vllm-wrapper`** — vLLM sleep-mode support. vLLM itself is not in the image.
- `llama-bench`, `llama-cli`, `llama-tts`, `whisper-cli`, `audiocpp_cli`.
- Upstream maintains the build logic; we maintain ~10 build args instead of ~200
  lines of cmake.

### What `ik-llama-server` is for

[`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) — "llama.cpp fork
with additional SOTA quants and improved performance" (3.1k stars). Same author as
llama.cpp's k-quants and i-quants. It adds the `IQ*_K` quant family and MoE-specific
optimisations that haven't been upstreamed.

Why we'd care: **`qwen3-coder-next:80b-a3b`**. An 80B MoE with ~3B active params is
precisely the shape ik_llama targets. We currently run it at `q4kxl` and `q5km`; the
IQK quants are the pitch for getting better quality at the same VRAM, or the same
quality smaller. It's a drop-in `llama-server` replacement in a model's `cmd:`, so it
can be A/B'd one model entry at a time against the stock binary.

### What `audiocpp_server` is for

[`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) — a pure C++/ggml audio
inference engine, **no Python dependency**. Covers TTS (50+ families), STT/ASR, VAD,
voice conversion, source separation, and music generation. OpenAI-compatible endpoints:
`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`, `/health`, plus
`/v1/tasks/run`.

Three concrete reasons it matters here:

1. **It supports Qwen3-TTS natively.** That's exactly what `/app/qwen3tts-venv` does
   today via Python. Moving `qwen3-tts:1.7b-clone` onto audio.cpp would let us delete
   *both* Python venvs — roughly **4 GB of torch + torchaudio** — and remove the
   torchaudio-cu130-caps-at-2.11.0 constraint from this Dockerfile entirely.
2. **It supports Qwen3-ASR**, which we previously ran as its own deployment
   (`manual/archive/qwen3-asr/`), and could consolidate the two `whisper-server` entries.
3. **It supports ACE-Step 1.5.** ACE-Step currently crashes under ComfyUI on our
   Blackwell cards because xformers has no sm_120 kernel for tensor `attn_bias`.
   audio.cpp is ggml, not PyTorch — no xformers involved. Worth trying as a way out
   of that dead end.

### TTS: what's dropped, what's kept, and why

**Orpheus is deleted.** A grep of the whole cluster repo (excluding `manual/archive`)
finds *zero* references to `/app/orpheus_server.py` or `/app/orpheus-venv` — no model
entry, no group member. It was a second full torch+torchaudio venv, ~4 GB, wired to
nothing. `orpheus_server.py` is not copied into this directory; it stays in git history
with the old image.

**Qwen3-TTS is kept, behind `INCLUDE_QWEN3TTS` (default `true`).** Not because it's
better than audio.cpp, but because the live entry is:

```yaml
"qwen3-tts:1.7b-clone":
  cmd: /app/qwen3tts-venv/bin/python /app/qwen3tts_server.py
       --mode clone --voices-dir /models/tts-voices
```

That's **zero-shot voice cloning** from reference audio (requires `ref_text`), using
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, and it's a `broadcaster-pipeline` member — broadcaster
prod *and* dev depend on it.

audio.cpp does list a `qwen3_tts` family, but its example config exposes **preset
voices** (`default_voice_preset.voice_id`), not zero-shot cloning. Parity on the clone
path is **unverified**. And the model on the PVC is HF-format
(`/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base` plus `/models/tts-voices`) while
audio.cpp is ggml and wants its own package layout — so migrating is a **model
conversion, not a config edit**.

Keep both present, A/B the clone output, then build `--build-arg INCLUDE_QWEN3TTS=false`
once audio.cpp is proven. That's when the torchaudio-2.11.0 ceiling leaves this file
for good.

Caveat: audio.cpp was created **2026-06-23** — about two months old and moving fast.
Mature enough to ship (v0.6.1, 437 commits, 2k stars) but treat it as the newest thing
in this image. It needs its own JSON model config; a starter is baked in at
`/etc/llama-swap/audiocpp-server.example.json`.

Changed:

- `llama-server` is now **statically linked** (`-DBUILD_SHARED_LIBS=OFF`). The old
  image used `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS` and shipped ggml backends
  as separate `.so` files. No more `LD_LIBRARY_PATH` juggling, but also no runtime
  CPU-variant selection.
- Binaries live in `/usr/local/bin`, not `/app`.
- llama-swap **v235 → v251**. This is a version bump riding along with the image
  change — if something misbehaves, rebuild with `--build-arg LLAMA_SWAP_VERSION=235`
  before blaming the unified build.

## Drop-in compatibility

Both configmaps hardcode `/app/...` paths — 30× `llama-server`, 4× `sd-server`,
2× `whisper-server`. The image creates `/app` symlinks for all three (plus `/app/sd`),
so **this image can be A/B tested by changing only the image tag**. No config edits.

Those symlinks are a migration aid. The long-term fix is a `macros:` block in the
configmaps (neither has one today):

```yaml
macros:
  "bindir": /usr/local/bin
models:
  "some-model":
    cmd: |
      ${bindir}/llama-server ...
```

Once that lands, delete the symlink `RUN` from the Dockerfile.

## Pinned versions and why

| component | pin | reason |
|---|---|---|
| CUDA | 13.2.1 | Driver 595.71.05 advertises 13.2 as its ceiling — not 13.3. sm_120 only needs ≥12.8, so this is parity, not a requirement. |
| `CUDA_ARCH` | 120 | The entire point. Also *faster* to build than upstream's 5-arch default. |
| llama-swap | v251 | Drives both the binary **and** the install scripts, so build logic and runtime always agree. |
| llama.cpp | b10595 | Bumped from b10434. That old pin was a floor, not a preference: `--reasoning-effort` does not exist before b10434, and on b10067 llama-server exited in ~800 ms with "upstream command exited prematurely". b10595 is well past it. Upstream's script defaults to `master`; we pin. |
| whisper.cpp | v1.9.3 | Bumped from v1.8.6. |
| stable-diffusion.cpp | `97d2990…` (full SHA) | Bumped from 90e87bc. Pinned by SHA because sd.cpp's `master-<N>-<hash>` tags aren't API-ordered. |
| ik_llama.cpp | `8337e4cd…` (full SHA) | No release tags upstream; SHA is the only way to pin. |
| audio.cpp | `62735eaf…` (full SHA) | Same. |
| torch / torchaudio | 2.11.0 | Re-verified 2026-08-23: torch cu130 ships up to 2.13.0 but **torchaudio cu130 still tops out at 2.11.0**. They must match, so 2.11.0 is still the ceiling. |
| qwen-tts, snac, fastapi, uvicorn, soundfile, requests, numpy | exact | Newly pinned — see below. |

### Newly pinned: the Python deps

The old image pinned torch/torchaudio but left everything else floating
(`pip install qwen-tts fastapi uvicorn[standard] soundfile`), so two builds of the
same Dockerfile could produce different images. They're now pinned to what PyPI
served on 2026-08-23, which is **what an unpinned build would have installed anyway** —
a reproducibility fix, not a version change.

## Everything is bumped — and what that costs

This image moves **every** axis at once, on purpose: it's a test vehicle, validated as
a unit. Production keeps the old pins until this passes. If it fails, bisect by
reverting ARGs one at a time.

| component | old (prod) | this image | drift |
|---|---|---|---|
| llama-swap | v235 | **v251** | 16 releases |
| llama.cpp | b10434 | **b10595** | 161 builds |
| whisper.cpp | v1.8.6 | **v1.9.3** | minor version |
| stable-diffusion.cpp | 90e87bc (2026-05-06) | **97d2990** (2026-08-19) | ~3.5 months |
| ik_llama.cpp | — | `8337e4cd…` | new |
| audio.cpp | — | `62735eaf…` | new |

Highest-risk items, in order:

1. **llama.cpp b10434 → b10595.** The old pin wasn't arbitrary — `--reasoning-effort`
   doesn't exist before b10434 and llama-server exited in ~800 ms on b10067. b10595 is
   far newer so the flag is there, but 161 builds is real drift. First diagnostic on
   any weirdness: `llama-server --help | grep reasoning`.
2. **sd.cpp, 3.5 months.** The old pin existed specifically to keep sd-server's CLI
   byte-for-byte compatible with our 4 `sd-server` entries. **Re-verify those flags.**
3. **whisper.cpp minor jump** — check the two `whisper-server` entries' flags.

`ik_llama.cpp` and `audio.cpp` have no release tags at all, so they're pinned by full
40-char commit SHA. Note `git fetch --depth=1` requires the *full* SHA — an
abbreviated one fails with `couldn't find remote ref`. sd.cpp is pinned the same way,
because its auto-generated `master-<N>-<hash>` tags aren't returned in chronological
order by the API and are genuinely hard to order by eye.

## Build

```bash
docker build \
  -t docker-private.privpub.krystof.io/krystof-io/llama-swap-unified-cuda13:latest .
```

Slim variant — no Python ML stack at all (~4 GB smaller, audio.cpp handles TTS):

```bash
docker build --build-arg INCLUDE_QWEN3TTS=false \
  -t docker-private.privpub.krystof.io/krystof-io/llama-swap-unified-cuda13:slim .
```

```
docker push docker-private.privpub.krystof.io/krystof-io/llama-swap-unified-cuda13:latest
```

This file uses `--mount=type=cache`, which needs BuildKit — but BuildKit has been
Docker's default builder since **23.0**, and we're on 29.7. No `DOCKER_BUILDKIT=1`
required; it would only matter on a pre-23 daemon.

Five CUDA projects compile from source. Expect a long first build; the ccache and
per-project build-cache mounts make rebuilds cheap.

## Testing plan

Test on `llama-swap-6000` first — it has only three models and nothing depends on it
yet. Leave the 5000 instance (broadcaster's backend) alone until the 6000 passes.

1. Build and push the image.
2. Point `manual/llama-swap/deployment-6000.yaml` at `:test`, apply, restart.
3. Verify:
   - pod starts, `/v1/models` lists all three
   - `nvidia-smi` inside the pod still shows only `GPU-2f289631` (the 6000)
   - the 9B task model loads and answers
   - the 27B loads; check load time against the ~16 s warm baseline — **a big
     regression here is the sm_120 JIT tell**
   - `/upstream/<model>/slots` still reports 4 slots / `n_ctx=262144`
   - 4 concurrent requests still overlap (~2 s, vs ~1 s single)
   - files written to `/models` are owned by `10001`, not root
4. Only then consider the 5000, which additionally exercises `sd-server`,
   `whisper-server` and the Qwen3-TTS venv via the broadcaster pipeline.

## Known things to watch

- **CUDA 13 + CDI runtime.** There is an open llama.cpp issue about CUDA 13 images
  failing to initialise under CDI when compat libs override the host driver. The
  `libcuda.so` stub here is deliberately kept in `lib64/stubs/` only and never on the
  default search path. The current production image is also CUDA 13.2 and works, so
  this is a watch item rather than a known break.
- **Orpheus is dead weight.** Nothing in either configmap references
  `/app/orpheus_server.py`. It is kept only so the A/B test is honest. Dropping that
  stage saves roughly 4 GB (a whole second torch + torchaudio).
- **`sd-server` health checks.** Upstream discussion #866 is *not* an image defect —
  `sd-server` has no `/health`, and llama-swap defaults `checkEndpoint` to `/health`.
  Our configs already set `checkEndpoint: "/"` on all four sd entries. A commenter
  there confirms `z-image-turbo` (one of ours) working on the unified image.

## Files

`qwen3tts_server.py`, `orpheus_server.py` and `config.example.yaml` are copies of the
ones in `../llama-swap-cuda13-with-extras/`. Keep them in sync while both images
exist; delete that directory once this one graduates.
