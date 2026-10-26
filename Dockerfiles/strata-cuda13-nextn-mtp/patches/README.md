# patches/

## 0001-nextn-mtp-blocks.patch

**What it's for:** Strata (any version up to and including v0.2.6, confirmed by reading the
current source) cannot load a `qwen4exp` GGUF whose metadata declares its own MTP/nextn
prediction block (`qwen4exp.nextn_predict_layers > 0`, `qwen4exp.block_count` = trunk + nextn).
Three independent paths fail: the architecture check rejects the file outright (block_count
doesn't match the compiled trunk size), the native-dense loader has no suffix entries for the
three `.nextn.*` 2-D projections so they hit a broken packed-fallback path, and `iq_pack.py`'s
own expert-table writer emits a row for the nextn block that the engine then refuses to load
("a malformed line") because its parser bounds-checks against a trunk-sized `n_layers`.

**Real model this hits:** `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`'s `IQ4_XS-MTP` quant
(`qwen4exp.block_count=49`, `nextn_predict_layers=1`) — confirmed crash-looping
`ghcr.io/niko1221/strata:0.2.3-cuda13` in production (exit 1, `EngineDied`, same error on every
restart).

**Where it comes from:** this is NOT an original diff. Niko1221/Strata contributors `mw00`
(PR #619) and `rluisr` (PR #1343, a rebase of #619 onto a later `main`) wrote, built, and
measured the real fix against a DIFFERENT model with the identical metadata shape
(`CYBER-FROST-3.8` / Blackfrost — same `block_count`/`nextn_predict_layers` pattern). Both PRs
were closed WITHOUT merging — the maintainer's reason was scope, not correctness:

> "Thank you. Model support is outside this speed pass." — Niko1221, closing PR #1343

Their own measurement from that PR: clean CUDA build, clean load, MTP drafter accepted 9,258 of
10,809 drafts (85.7%) over 274 drafting windows. The code is real and tested; it just never
shipped.

**Why this patch file is not just `curl`'d from the PR:** `https://github.com/Niko1221/Strata/
commit/94a3ae2.diff` (the commit both PRs point at) does NOT `git apply` cleanly against v0.2.6
or current `main` — confirmed by trying both. The surrounding code has been refactored since
that commit: the `Qwen4ExpGuard` struct the original patch modified no longer exists; the
architecture check is now split across `read_geometry()` / `geometry_unsupported()` /
`check_architecture()`, and a newer model family (`qwen35moe`) already has its OWN
`nextn_predict_layers`-aware check (`check_architecture_qwen35moe()`) that this patch's
`gguf_reader.hpp` change deliberately mirrors for `qwen4exp`.

This patch is the SAME three changes (same files, same intent, same reasoning in the original
PR description), hand-re-applied against v0.2.6's current code shape:

1. `include/strata/artifact/gguf_reader.hpp` — `read_geometry()` reads
   `qwen4exp.nextn_predict_layers` (absent → 0) and subtracts it from `g.n_layers` before the
   trunk-size comparison, mirroring `check_architecture_qwen35moe()`.
2. `src/core/native_dense.cpp` — `eligible()` gains the three `.nextn.*` 2-D projection suffixes.
3. `tools/iq_pack.py` — `expert_layout()` subtracts the declared `nextn` count from `n_layers` so
   `native_experts.txt` covers the trunk only.

**Safety for every other model:** a GGUF without `qwen4exp.nextn_predict_layers` reads it as
absent → `nextn = 0` → every new code path is a no-op (subtracting 0, an empty suffix-list
addition that never matches, `n_layers - 0`). This is additive; it does not change how Q2_0,
Orca IQ2_M, or any other currently-working model loads. Not yet re-verified end-to-end against
those models on THIS build (see "Known gaps" below) — the reasoning is sound and traces the
same no-op path the original PR's own description claims, but hasn't been re-measured here.

**Verify it still applies before building**, same discipline the original PR data used — a
newer upstream release may have moved these functions again:

```bash
cd /tmp && git clone --depth 1 --branch v0.2.6 https://github.com/Niko1221/Strata.git strata-verify
cd strata-verify && git apply --check /path/to/0001-nextn-mtp-blocks.patch
```

If it fails: diff the 3 touched functions (`read_geometry`, `eligible` in `native_dense.cpp`,
`expert_layout` in `iq_pack.py`) between the pinned `STRATA_VERSION` and whatever the patch was
last verified against (recorded in the Dockerfile's version-pinning comment block), and
hand-reapply the same 3 changes to the new shape — same approach used to create this patch in
the first place.

### Known gaps — what has NOT been verified yet

- **Build and real-model-load testing completed on 2026-10-26.** First build of this image
  (against the original 3-change patch) exposed TWO real bugs, not one — see below.
- **Real bug #2, found and fixed during the first live load attempt**: subtracting `nextn`
  from `g.n_layers` in `read_geometry()` is correct for the trunk-size comparison, but it also
  shrinks the size `qwen4exp.attention.compress_ratios` is checked against — and this GGUF's
  array is sized to the RAW `block_count` (49), not the reduced trunk (48), producing a second,
  different rejection: `qwen4exp.attention.compress_ratios must list one entry for each of the
  48 layers`. Fixed by accepting EITHER size (`g.n_layers` or `g.n_layers + nextn`) in that one
  check — every loop that reads the array already only touches the first `g.n_layers` (trunk)
  entries, so no other code needed to change. This specific model's `compress_ratios` turned out
  to be all-zeros anyway (a known placeholder shape per the pre-existing `#2396` comment this
  function already carried for orcarouter GGUFs), so the fix's effect here is just "stop
  rejecting the file before reaching that all-zeros short-circuit" — not yet proven against a
  model whose `compress_ratios` carries real non-zero values at the reduced trunk size.
- **End-to-end real load of `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`'s `IQ4_XS-MTP`
  quant: in progress as of this commit** — the "malformed line" bug is confirmed gone (pack
  rebuild produces 48 rows, 0 `blk.48` references, matching `CYBER_FROST.md`'s own documented
  verification exactly) and the compress_ratios bug above is fixed; whether the engine reaches
  a real `READY` state and serves a coherent completion with this model's embedded MTP head is
  the next thing to confirm, not yet in hand as of this patch version.
- **Regression-tested against the known-good Orca IQ2_M model**: confirmed clean — real
  completion returned, MTP speculative decoding active (`draft_n_accepted` present), vision
  active (`modalities.vision: true`), on this exact image. This is real evidence the patch is a
  no-op for a model without an MTP/nextn block, not just the originally-reasoned claim.
