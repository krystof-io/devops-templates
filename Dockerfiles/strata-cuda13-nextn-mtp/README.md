# strata-cuda13-nextn-mtp

Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata)), built from source at `v0.2.6`
with one local patch applied: engine support for a `qwen4exp` GGUF that declares its own
MTP/nextn prediction block (`qwen4exp.nextn_predict_layers > 0`).

See the Dockerfile's own header comment and `patches/README.md` for the full why/what. Short
version: the published `ghcr.io/niko1221/strata:0.2.3-cuda13` image we run in
`manual/strata/` cannot load `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`'s `IQ4_XS-MTP`
quant — it crash-loops with `native_experts.txt: a malformed line`. The fix is a real,
previously-written-and-measured patch from two unmerged upstream PRs (#619, #1343), rejected by
the maintainer on scope grounds ("Model support is outside this speed pass"), not correctness.
This image carries that fix, hand-re-applied against v0.2.6's current code shape since the
original diff no longer `git apply`s cleanly.

## Build

```bash
cd Dockerfiles/strata-cuda13-nextn-mtp
docker build -t docker-private.privpub.krystof.io/krystof-io/strata-cuda13-nextn-mtp:0.2.6 .
docker push  docker-private.privpub.krystof.io/krystof-io/strata-cuda13-nextn-mtp:0.2.6
```

`CUDA_ARCHITECTURES` defaults to `120` (RTX PRO 6000 / 5000 Blackwell, our only cards) — much
faster than upstream's own 5-arch default build. Pass `--build-arg CUDA_ARCHITECTURES="86;89;120"`
etc. if building for other hardware.

No GPU is needed to build — same as upstream's own `Dockerfile`, the CUDA architectures are
given explicitly rather than probed from `nvidia-smi`.

## Status

**Not yet built or tested as of this commit.** The patch applies cleanly (`git apply --check`
against a fresh `v0.2.6` clone, confirmed) but has not yet been compiled or run against a real
model. See `patches/README.md`'s "Known gaps" section before trusting this beyond "the source
diff is plausible."

## Using it

Same runtime image shape as upstream's own `ghcr.io/niko1221/strata` (`/data` volume,
`entrypoint.sh`, port 8080, env-var or direct `serve.server` launch) — see `manual/strata/` in
`kio-public-cluster` for the live Kubernetes Deployment this is meant to replace once verified.
Change only the `image:` tag to cut over; no other manifest changes are expected.
