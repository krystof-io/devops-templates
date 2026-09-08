# Local patches to pinned upstream sources

Patches in this directory are applied to `audio.cpp` after `install-audio.sh`
has cloned and built it, and the affected binary is then rebuilt incrementally.
See the `audio-build` stage in the Dockerfile for the mechanics and for why the
patches are applied *after* the upstream build rather than before.

Every patch here is a liability: it is a local fork of a pinned third-party
commit. Each one should be sent upstream, and each one must be re-verified when
its pin moves. `git apply --check` runs before the real apply, so a patch that
no longer applies **fails the build** rather than silently producing an
unpatched binary.

---

## voxcpm2-streaming-seam.patch

**Fixes:** audible clicks at every chunk boundary in VoxCPM2 streaming output.

**Pin it applies to:** `AUDIO_CPP_VERSION=62735eafd96294c52d6c4607f5f38ac55be54f06`

**File touched:** `src/models/voxcpm2/session.cpp`, the `emit_chunk` lambda in
`VoxCPM2SessionBase::run_streaming_request`.

### The bug

The streaming session decodes one patch at a time via
`decode_features(patch, 1)`. That call is stateless: it zero-fills its input
buffer, runs the graph, reads the samples back. So every chunk restarts the
AudioVAE's causal convolutions from silence, the leading samples of each chunk
are decoded without their history, and each boundary is audible.

Measured on an 8 s utterance at 48 kHz through the deployed model: 19 sample
discontinuities, every one landing on a multiple of the 0.16 s patch period
(`patch_size` 4 x decoder stride 7,680 samples). The offline session, which
decodes the whole utterance in one graph, produced zero from the same weights,
seed and text. It is not the quantisation and not a sampling setting.

### The fix

Keep the last three decoded patches, decode them together with the current one,
and erase their samples before emitting. Every decoder convolution is causal, so
once the prefix covers the receptive field the retained tail is exactly what a
whole-utterance decode produces.

This is not a new technique in this codebase: `run_offline_request`, in the same
file, already does prefix-and-trim using `generated.decode_trim_patches`. Three
patches of context matches the reference implementation, whose
`streaming_prefix_len` of 4 is three patches plus the current one. OpenBMB later
replaced that overlap decode with cached convolution state, which is the same
idea done more cheaply.

Cost is decoder work only, four patches per chunk instead of one. The language
model and diffusion work per chunk is unchanged and dominates, so time to first
audio is unaffected. The merged whole-utterance buffer is built from the same
trimmed chunks, so plain non-streaming requests to a streaming-mode model are
fixed too.

### Verified result

Built and deployed 2026-09-07, image `sha256:9bb21d7f…`, measured on the same
8 s jocasta utterance used to find the bug:

| | before | after |
|---|---|---|
| streamed, discontinuities on the 0.16 s grid | 19 | 0 |
| plain WAV from the streaming-mode model | 19 | 0 |
| time to first audio chunk | 240-270 ms | 221 ms |
| resident VRAM | 7.8 GB | 7.8 GB |

Against the offline session at the same seed the patched streaming output is
not bit-identical, and that is expected: three patches of prefix do not cover
the decoder's full receptive field. The residual sits 54.8 dB below the signal
and is spread evenly rather than clustered at chunk boundaries, 1.12x inside a
4 ms window around a boundary versus outside, so there is no residual seam,
only the receptive-field tail. Raising `kStreamingDecodeContextPatches` is the
lever if that ever proves audible.

### How to verify after a build

The clicks are periodic, so the check is whether discontinuities land on the
patch grid. Anything above zero grid hits means the patch did not take effect.

```bash
# Stream from the patched model and save raw PCM
curl -sN https://llm-6000.public.krystof.io/v1/audio/speech \
  -H 'Content-Type: application/json' -H 'Accept-Encoding: identity' \
  -d '{"model":"voxcpm2:audiocpp-clone","input":"The quick brown fox jumped over the lazy dog. It was a beautiful morning and the sun was shining brightly through the trees.","voice":"jocasta","response_format":"pcm","stream_format":"audio"}' \
  > vox.pcm
sox -t raw -r 48000 -e signed -b 16 -c 1 vox.pcm vox.wav

# Count discontinuities that land on the 0.16 s patch grid
python3 - vox.wav <<'EOF'
import sys, wave, struct, statistics
w = wave.open(sys.argv[1]); sr = w.getframerate(); n = w.getnframes()
x = struct.unpack('<%dh' % n, w.readframes(n))
d = [abs(x[i] - x[i-1]) for i in range(1, n)]
thr = max(1500, 40 * statistics.median(d))
pts, merged = [i for i, v in enumerate(d) if v > thr], []
for i in pts:
    if merged and i - merged[-1] < sr * 0.005: continue
    merged.append(i)
patch = int(0.16 * sr)
on = [i for i in merged if min(i % patch, patch - i % patch) <= int(0.001 * sr)]
print('on-grid discontinuities: %d  (expected 0 with the patch, ~19 without)' % len(on))
EOF
```

Confirm the binary actually carries the patch:

```bash
kubectl exec -n llama-swap deploy/llama-swap-6000 -c llama-swap -- \
  sh -c 'strings /usr/local/bin/audiocpp_server | grep "streaming decode context trim"'
# The patch adds this error string; an unpatched binary prints nothing.
```

Also worth a listen rather than only a metric: the failure this fixes is
perceptual, and the grid count cannot tell you the prefix was long enough, only
that boundaries stopped being discontinuous. If clicks persist while the grid
count reads zero, raise `kStreamingDecodeContextPatches` in the patch.

### Build-cache caveat

The patch step reuses the build directory that `install-audio.sh` populated,
which lives in a BuildKit cache mount. If that cache has been pruned while the
first `RUN` stays layer-cached, `cmake --build build` fails because the build
directory is not configured. The fix is to force the upstream build to run
again:

```bash
docker build --no-cache-filter audio-build -t <image> .
```

### Upstream

**Deliberately not filed** (decided 2026-09-07). This stays a local patch; do
not treat the absence of a PR as an oversight to correct.

The consequence is that it will not arrive on its own. When `AUDIO_CPP_VERSION`
moves, check whether upstream fixed this independently before rebasing the
patch. As of the current pin, nothing touching `src/models/voxcpm2` had landed
since it except a UI chunking change. Upstream is
https://github.com/0xShug0/audio.cpp, and the case for filing it, if that is
ever reconsidered, is that `run_offline_request` in the same file already
contains the technique, so this reads as a gap in the port rather than a design
decision.
