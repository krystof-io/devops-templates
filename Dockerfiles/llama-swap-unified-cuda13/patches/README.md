# Local patches to pinned upstream sources

Patches in this directory are applied to their respective upstream project
after that project's own install script has cloned and built it, and the
affected binary is then rebuilt incrementally. See the matching build stage
in the Dockerfile for the mechanics and for why the patches are applied
*after* the upstream build rather than before.

Every patch here is a liability: it is a local fork of a pinned third-party
commit. Each one should be sent upstream, and each one must be re-verified
when its pin moves. `git apply --check` runs before the real apply, so a
patch that no longer applies **fails the build** rather than silently
producing an unpatched binary.

---

## sdcpp-lokr-multidim-batch.patch

**Fixes:** `GGML_ASSERT`/CUDA crash applying a LoKr-format LoRA (`lokr_w1`/
`lokr_w2` factors) to any model whose attention feeds linears a 3-D
activation `[features, L, N]` with `N > 1` — confirmed live on Krea-2-Turbo
(`krea-2-turbo:12b-q8_0` on llama-swap-6000) with a real LoKr LoRA
(`snofs_krea_v1_5.safetensors`).

**Pin it applies to:** `SD_CPP_VERSION=f89d9b13d730eabeede7314ce49dacd18d3c90c2`
(verified `git apply --check` clean against that exact commit; the fix diff
is unmodified from the withdrawn upstream PR).

**File touched:** `src/model/adapter/lora_ops.cpp`, `ggml_ext_lokr_forward`'s
non-conv branch.

### The bug

`ggml_ext_lokr_forward` assumes a 2-D activation `[q, batch]` and takes
`batch = (int)h->ne[1]`. Krea2's `KreaAttention` (and other batched
attention paths) feed linears a 3-D activation `[features, L, N]` instead.
Whenever `N > 1`, `ggml_nelements(h) = q·L·N ≠ vq·uq·L`, so the split
reshape's element-count `GGML_ASSERT` fires and the process aborts. With
`N == 1` every reshape happens to pass, which is why LoKr works fine on many
models and this gap went unnoticed until a real multi-batch request hit it.

Confirmed live with TWO distinct crash signatures depending on
`--lora-apply-mode`:
- `at_runtime` (the default for a quantized model): CPU-side
  `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)` in `ggml_reshape_4d`,
  called from `ggml_ext_lokr_forward` → `LoraModel::get_out_diff` →
  `Linear::forward` → `Krea2::KreaAttention::forward`.
- `immediately`: a CUDA `cublasSgemm_v2` launch failure in
  `ModelManager::apply_loras_to_params`, merging the same malformed shapes
  at load time instead of at runtime. Different failure mode, same root
  cause.

Both resolved with `addr2line` against the actual deployed binary, not
guessed from source reading.

### The fix

- `batch = ggml_nelements(h) / q_actual` — fold ALL trailing dims into the
  batch (the existing reshapes already assume a contiguous activation), not
  just `ne[1]`.
- Reshape the result back to `[up·vp, ne[1], ne[2], ne[3]]` instead of flat
  2-D, so the diff adds element-wise onto the linear output rather than
  depending on a shape that no longer matches once the batch is folded
  correctly.
- The conv branch already batches on `ne[3]` and is unchanged.

### Verified result

Built and deployed 2026-10-10 (image tag TBD on actual build). Before the
patch: 100% crash on any `lora` array request against `krea-2-turbo:12b-q8_0`
with the LoKr LoRA present, both apply modes. After: TBD — fill in once the
rebuilt image is actually tested end to end (first real render with the
LoRA applied, checked for both success and visual correctness, the same
bar as every other verified-working note in this repo).

### Upstream status

Originally submitted as leejet/stable-diffusion.cpp#1980 by
`undefeaterpeter`. **Withdrawn by its own author** with no stated reason
(the only comment on the PR is the single word "Withdrawn."). As of
`f89d9b13d730eabeede7314ce49dacd18d3c90c2` (2026-10-09, current master at
the time this patch was written), the exact line the PR identified as
broken — `int batch = (int)h->ne[1];` — is still present verbatim. Nobody
else has fixed this since.

**This stays a local patch**, same posture as the (now-removed) VoxCPM2
patch used to have: filing fresh against upstream is reasonable since it's
a real, cleanly-diagnosed bug with a working fix already written by someone
else, but don't expect it to land on its own — re-check whether upstream
(or a differently-shaped fix) addressed this independently before rebasing,
the same way the VoxCPM2 patch below turned out to be supersedable.

### How to verify after a build

```bash
# Confirm the binary carries the patch — the folded-batch computation has no
# distinguishing string to grep for, so verify behaviorally instead:
# a LoKr LoRA request against a model with batched (N>1) attention activations
# must NOT crash. Warm the LoRA registry first (lazy-scanned):
curl -s "https://llm-6000.public.krystof.io/sdapi/v1/loras?model=krea-2-turbo:12b-q8_0"
curl -s -X POST https://llm-6000.public.krystof.io/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"model":"krea-2-turbo:12b-q8_0","prompt":"a cat balancing on a basketball",
       "width":1024,"height":1024,"steps":8,"cfg_scale":1.0,"seed":7777,
       "sampler_name":"euler","scheduler":"discrete",
       "lora":[{"path":"snofs_krea_v1_5.safetensors","multiplier":1.0}]}' \
  -o /tmp/lokr_test.json -w "HTTP_CODE:%{http_code}\n"
# Expect 200, not 500/502. Then actually decode and look at the image —
# a 200 with a crashed/garbled render is not a pass either.
```

### Build-cache caveat

Same as the audio.cpp pattern: the patch step reuses the `sd-cuda13` build
cache mount that `install-sd.sh` populated. If that cache was pruned while
the first `RUN` stays layer-cached, `cmake --build build` fails because the
build directory is not configured. Force the upstream build to re-run with
`docker build --no-cache-filter sd-build -t <image> .`.

---

## voxcpm2-streaming-seam.patch — REMOVED 2026-10-10, kept here as history

**Was:** audible clicks at every chunk boundary in VoxCPM2 streaming output,
fixed by decoding the last three patches as rolling context and trimming
them before emitting (the `kStreamingDecodeContextPatches` technique).

**Why removed:** upstream fixed the identical bug independently, 13 days
after this patch was written, with the same technique — commits
`a045e690a6` ("decode streaming patches with left context") and
`6c70f32d0d` ("correct the streaming decode rationale, drop right context"),
both 2026-09-20. Upstream's `stream_left_context` request option defaults to
`3`, matching this patch's own `kStreamingDecodeContextPatches = 3`.

Confirmed removable, not just supersedable, by actually trying: the old
patch's `git apply --check` **fails** against
`AUDIO_CPP_VERSION=715108a8efde4bb9831dd19771dc3a5efbbdc953` (2026-10-09) —
upstream's own session.cpp has moved on from the exact code shape this patch
targeted. This is the scenario patches/README.md always said to check for
before rebasing a stale patch, and this time the answer was "don't bother,
it's native now." No `--request-option` change needed on our side: the
upstream default (3) already matches what we were already getting.

The original patch text and full writeup are preserved in git history on
this file (see the commit that added this removal note) in case
`stream_left_context` is ever retuned and the original measurement data is
useful context.
