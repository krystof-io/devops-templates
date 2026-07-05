# llama-swap-branch-cuda13-with-extras

[llama-swap](https://github.com/mostlygeek/llama-swap) bundled with extra
CUDA-accelerated inference servers, built for the upgraded GPU node
(**worker-4**: driver `595.71.05`, CUDA 13.2 ceiling). Built as a multi-arch
fatbinary for both Ada (RTX 4070 Ti Super / `sm_89`) and Blackwell
(RTX 50-series / `sm_120`).

Fork of `llama-swap-cuda13-with-extras` with Qwen3.6 hybrid-arch cache patches
applied at build time (checkpoint search fix + recurrent shrink/expand API).
See Dockerfile for patch details.

Unlike `llama-swap-cuda12-with-extras` — which layered onto the prebuilt
`mostlygeek/llama-swap:cuda` image (still pinned to **CUDA 12.8.1** upstream) —
this image builds `llama-server`, `whisper-server` and `sd-server` **all from
source on a single CUDA 13.2 base**, so the entire image is CUDA-13 consistent.
It then reproduces the upstream llama-swap runtime layout (everything under
`/app`, non-root `uid 10001`, `/app` on `PATH`, llama-swap entrypoint).

## Bundled binaries (all under `/app`, on `PATH`)

| Binary | Source | Purpose |
|--------|--------|---------|
| `llama-swap` | mostlygeek/llama-swap v223 | model-swapping proxy / entrypoint |
| `llama-server` | ggml-org/llama.cpp fdb1db877c (patched) | LLM inference (CUDA 13.2) |
| `whisper-server` | ggml-org/whisper.cpp v1.8.6 | speech-to-text (CUDA 13.2) |
| `sd-server` | leejet/stable-diffusion.cpp master-596-90e87bc | image generation (CUDA 13.2) |
| `qwen3tts_server.py` | Qwen3-TTS (`/app/qwen3tts-venv`) | text-to-speech (torch 2.11.0 +cu130) |
| `orpheus_server.py` | Orpheus TTS + SNAC (`/app/orpheus-venv`) | text-to-speech (torch 2.11.0 +cu130) |

## Manual build and push

Log in to the registry:

```bash
docker login <repohost>
```

Build (no GPU required — CUDA arch is set via build-arg):

```bash
docker build --progress=plain \
  --build-arg CUDA_VERSION="13.2.1" \
  --build-arg CUDA_ARCH="89;120" \
  -t <repohost>/krystof-io/llama-swap-branch-cuda13-with-extras:latest \
  .
```

Push:

```bash
docker push <repohost>/krystof-io/llama-swap-branch-cuda13-with-extras:latest
```

## Notes

- Mount your real llama-swap config over `/app/config.yaml`; the baked-in
  `config.example.yaml` is just a placeholder so the entrypoint starts.
- **GPU arch:** `CUDA_ARCH` is a semicolon-separated list of compute
  capabilities baked into the binaries. `89;120` covers Ada *and* Blackwell, so
  the same image runs unchanged when worker-4's GPU is swapped to a 50-series
  card. The PyTorch cu130 wheels already ship Blackwell kernels. If you add a
  GPU with a different capability, add its number here and rebuild.
- **stable-diffusion.cpp is intentionally *not* on its newest tag.** Its
  auto-generated tags aren't returned chronologically by the GitHub API, and
  some recent ones drop the server example entirely. It's pinned to the exact
  version the cuda12 image runs (`master-596-90e87bc`) so `sd-server`'s CLI
  stays config-compatible — we only recompile it on CUDA 13. Bump it
  deliberately (and re-test the sd flags) if you want newer sd features.
- Component versions are pinned via build-args at the top of the `Dockerfile`
  (`LLAMA_SWAP_VERSION`, `WHISPER_CPP_VERSION`, `SD_CPP_VERSION`). llama.cpp is
  pinned to commit `fdb1db877c` (2026-07-02) for the Qwen3.6 hybrid-arch patches —
  bump deliberately and verify the patches still apply.
- `torch`/`torchaudio` are pinned to `2.11.0 +cu130` (the highest version where
  both have CUDA-13.0 wheels) and installed via `--index-url` so PyPI's
  CUDA-13.0 torch can't shadow the cu130 wheel.
- CUDA toolkit images go up to 13.3, but the driver advertises **13.2** as its
  ceiling, so the toolkit is pinned to 13.2.1 to avoid a runtime-version
  mismatch.
