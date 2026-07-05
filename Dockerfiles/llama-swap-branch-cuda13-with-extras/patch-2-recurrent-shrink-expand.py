#!/usr/bin/env python3
"""
PATCH 2: Recurrent shrink/expand API for prompt cache operations.

Shrinks recurrent state to 1 cell before prompt cache save/load, then expands
back. Prevents forced full re-processing on hybrid architectures (Qwen3.6).
Based on upstream PR #24785.

Modifies:
  - include/llama.h              (C API declarations)
  - src/llama-context.h          (resize_recurrent_memory declaration)
  - src/llama-context.cpp        (includes, method impl, C API wrappers)
  - src/llama-memory-recurrent.h (expand/shrink/resize declarations)
  - src/llama-memory-recurrent.cpp (expand/shrink/resize implementation)
  - tools/server/server-context.cpp (recurrent model tracking, shrink/expand hooks)
"""
import sys


def patch_file(path, replacements):
    """Apply a list of (search, insert_after, text) replacements."""
    with open(path, "r") as f:
        src = f.read()
    for i, (search, insert_after, text) in enumerate(replacements):
        pos = src.find(search)
        if pos == -1:
            print(f"PATCH 2.{i} FAILED: search string not found in {path}", file=sys.stderr)
            sys.exit(1)
        anchor_end = pos + len(search)
        if insert_after:
            after_pos = src.find(insert_after, anchor_end)
            if after_pos == -1:
                print(f"PATCH 2.{i} FAILED: insert_after not found in {path}", file=sys.stderr)
                sys.exit(1)
            anchor_end = after_pos + len(insert_after)
        src = src[:anchor_end] + text + src[anchor_end:]
    with open(path, "w") as f:
        f.write(src)


# ---------------------------------------------------------------------------
# 1. llama.h — Add recurrent expand/shrink C API declarations
# ---------------------------------------------------------------------------
patch_file("include/llama.h", [
    (
        "LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);",
        None,
        """

    // Expand the recurrent state to new_n_seq_max cells (for deferred backup allocation).
    // Returns true on success. No-op if the memory is already large enough or has no recurrent component.
    LLAMA_API bool llama_memory_recurrent_expand(llama_memory_t mem, uint32_t new_n_seq_max);

    // Shrink the recurrent state to new_n_seq_max cells (frees GPU memory for prefill).
    // Returns true on success. No-op if the memory is already small enough or has no recurrent component.
    LLAMA_API bool llama_memory_recurrent_shrink(llama_memory_t mem, uint32_t new_n_seq_max);

    // Context-level recurrent resize. These variants also invalidate the context scheduler/graph cache
    // because recurrent tensors are reallocated and graph nodes hold tensor pointers.
    LLAMA_API bool llama_context_recurrent_expand(struct llama_context * ctx, uint32_t new_n_seq_max);
    LLAMA_API bool llama_context_recurrent_shrink(struct llama_context * ctx, uint32_t new_n_seq_max);"""
    ),
])

# ---------------------------------------------------------------------------
# 2. llama-context.h — Add resize_recurrent_memory declaration
# ---------------------------------------------------------------------------
patch_file("src/llama-context.h", [
    (
        "void set_warmup(bool value);",
        None,
        """

    bool resize_recurrent_memory(uint32_t new_n_seq_max, bool expand);"""
    ),
])

# ---------------------------------------------------------------------------
# 3. llama-context.cpp — Includes + resize_recurrent_memory impl + C API
# ---------------------------------------------------------------------------

# 3a. Add includes after #include "llama-memory.h"
patch_file("src/llama-context.cpp", [
    (
        '#include "llama-memory.h"',
        None,
        '''
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"'''
    ),
])

# 3b. Add resize_recurrent_memory method after set_warmup body.
#     Anchor on the full closing of set_warmup (comment + brace + blank line + next function)
#     to avoid inserting inside set_warmup's body.
path = "src/llama-context.cpp"
with open(path, "r") as f:
    src = f.read()

set_warmup_close = """    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler"""

if set_warmup_close not in src:
    print("PATCH 2.3b FAILED: could not find set_warmup closing pattern", file=sys.stderr)
    sys.exit(1)

resize_method = """    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::resize_recurrent_memory(uint32_t new_n_seq_max, bool expand) {
    if (!memory) {
        return false;
    }

    auto * recr = dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (!recr) {
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
        if (hybrid) {
            recr = hybrid->get_mem_recr();
        } else {
            auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory.get());
            if (hybrid_iswa) {
                recr = hybrid_iswa->get_mem_recr();
            }
        }
    }
    if (!recr) {
        return true; // no recurrent component - nothing to resize
    }

    synchronize();

    const bool ok = expand ? recr->expand(new_n_seq_max) : recr->shrink(new_n_seq_max);
    if (ok) {
        sched_need_reserve = true;
        if (gf_res_prev) {
            gf_res_prev->reset();
        }
    }

    return ok;
}

bool llama_context::set_sampler"""

src = src.replace(set_warmup_close, resize_method, 1)
with open(path, "w") as f:
    f.write(src)

# 3c. Add C API functions after llama_memory_can_shift body.
#     Anchor on the stable closing pattern: return + brace + blank line + comment.
C_API_FUNCS = """

bool llama_memory_recurrent_expand(llama_memory_t mem, uint32_t new_n_seq_max) {
    if (!mem) return false;
    auto * recr = dynamic_cast<llama_memory_recurrent *>(mem);
    if (!recr) {
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
        if (hybrid) recr = hybrid->get_mem_recr();
        else {
            auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem);
            if (hybrid_iswa) recr = hybrid_iswa->get_mem_recr();
        }
    }
    return recr ? recr->expand(new_n_seq_max) : true;
}

bool llama_memory_recurrent_shrink(llama_memory_t mem, uint32_t new_n_seq_max) {
    if (!mem) return false;
    auto * recr = dynamic_cast<llama_memory_recurrent *>(mem);
    if (!recr) {
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
        if (hybrid) recr = hybrid->get_mem_recr();
        else {
            auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem);
            if (hybrid_iswa) recr = hybrid_iswa->get_mem_recr();
        }
    }
    return recr ? recr->shrink(new_n_seq_max) : true;
}

bool llama_context_recurrent_expand(llama_context * ctx, uint32_t new_n_seq_max) {
    return ctx ? ctx->resize_recurrent_memory(new_n_seq_max, true) : false;
}

bool llama_context_recurrent_shrink(llama_context * ctx, uint32_t new_n_seq_max) {
    return ctx ? ctx->resize_recurrent_memory(new_n_seq_max, false) : false;
}"""

anchor = """    return mem->get_can_shift();
}

// llama state API"""
if anchor not in src:
    print("PATCH 2.3c FAILED: could not find llama_memory_can_shift closing pattern", file=sys.stderr)
    sys.exit(1)

replacement = """    return mem->get_can_shift();
}""" + C_API_FUNCS + """

// llama state API"""
src = src.replace(anchor, replacement, 1)
with open(path, "w") as f:
    f.write(src)

# ---------------------------------------------------------------------------
# 4. llama-memory-recurrent.h — Add expand/shrink/resize declarations
# ---------------------------------------------------------------------------
patch_file("src/llama-memory-recurrent.h", [
    (
        "bool get_can_shift() const override;",
        None,
        """

    // Expand/shrink the recurrent state memory (for prompt cache save/restore)
    bool expand(uint32_t new_mem_size);
    bool shrink(uint32_t new_mem_size);"""
    ),
    (
        "std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;",
        None,
        """

    bool resize(uint32_t new_mem_size);""",
    ),
])

# ---------------------------------------------------------------------------
# 5. llama-memory-recurrent.cpp — Add expand/shrink/resize implementation
# ---------------------------------------------------------------------------
RESIZE_IMPL = """

bool llama_memory_recurrent::expand(uint32_t new_mem_size) {
    return new_mem_size <= size || resize(new_mem_size);
}

bool llama_memory_recurrent::shrink(uint32_t new_mem_size) {
    return new_mem_size >= size || resize(new_mem_size);
}

bool llama_memory_recurrent::resize(uint32_t new_mem_size) {
    if (new_mem_size == size) {
        return true;
    }

    const int32_t n_layer = hparams.n_layer();
    const uint32_t old_size = size;
    const uint32_t n_copy = std::min(old_size, new_mem_size);

    struct buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u * n_layer * ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) {
                return nullptr;
            }
            ctx_map.emplace(buft, std::move(ctx));
            return ctx_map.at(buft).get();
        }
        return it->second.get();
    };

    std::vector<ggml_tensor *> old_r_l = r_l;
    std::vector<ggml_tensor *> old_s_l = s_l;

    for (int i = 0; i < n_layer; i++) {
        if (!old_r_l[i] && !old_s_l[i]) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(old_r_l[i] ? old_r_l[i]->buffer : old_s_l[i]->buffer);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for resized rs cache\\n", __func__);
            return false;
        }

        if (old_r_l[i]) {
            ggml_tensor * r = ggml_new_tensor_2d(ctx, old_r_l[i]->type, hparams.n_embd_r(), new_mem_size * (1 + n_rs_seq));
            ggml_format_name(r, "cache_r_l%d", i);
            r_l[i] = r;
        }
        if (old_s_l[i]) {
            ggml_tensor * s = ggml_new_tensor_2d(ctx, old_s_l[i]->type, hparams.n_embd_s(), new_mem_size * (1 + n_rs_seq));
            ggml_format_name(s, "cache_s_l%d", i);
            s_l[i] = s;
        }
    }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> new_ctxs_bufs;
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate resized rs buffer\\n", __func__);
            r_l = old_r_l;
            s_l = old_s_l;
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        new_ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    if (n_copy > 0) {
        const uint32_t n_copy_rows = n_copy * (1 + n_rs_seq);
        std::vector<uint8_t> tmp;
        for (int i = 0; i < n_layer; i++) {
            if (old_r_l[i] && r_l[i]) {
                size_t bytes = ggml_row_size(old_r_l[i]->type, hparams.n_embd_r()) * n_copy_rows;
                tmp.resize(bytes);
                ggml_backend_tensor_get(old_r_l[i], tmp.data(), 0, bytes);
                ggml_backend_tensor_set(r_l[i], tmp.data(), 0, bytes);
            }
            if (old_s_l[i] && s_l[i]) {
                size_t bytes = ggml_row_size(old_s_l[i]->type, hparams.n_embd_s()) * n_copy_rows;
                tmp.resize(bytes);
                ggml_backend_tensor_get(old_s_l[i], tmp.data(), 0, bytes);
                ggml_backend_tensor_set(s_l[i], tmp.data(), 0, bytes);
            }
        }
    }

    ctxs_bufs = std::move(new_ctxs_bufs);
    cells.resize(new_mem_size);
    size = new_mem_size;

    uint32_t used_new = 0;
    for (auto & cell : cells) {
        cell.tail = -1;

        for (auto it = cell.seq_id.begin(); it != cell.seq_id.end();) {
            if (*it < 0 || (uint32_t) *it >= size) {
                LLAMA_LOG_WARN("%s: dropping seq_id %d after resize %u -> %u\\n",
                        __func__, *it, old_size, new_mem_size);
                it = cell.seq_id.erase(it);
            } else {
                ++it;
            }
        }

        if (cell.seq_id.empty()) {
            cell.pos  = -1;
            cell.src  = -1;
            cell.src0 = -1;
            continue;
        }

        cell.src = -1;
        cell.src0 = -1;

        ++used_new;
    }

    used = used_new;

    if (head >= size) {
        head = 0;
    }
    if (n >= size) {
        n = 0;
    }

    return true;
}"""

patch_file("src/llama-memory-recurrent.cpp", [
    (
        "bool llama_memory_recurrent::get_can_shift() const {\n    // shifting the pos is trivial for recurrent models\n    return true;\n}",
        None,
        RESIZE_IMPL,
    ),
])

# ---------------------------------------------------------------------------
# 6. server-context.cpp — Recurrent model tracking + shrink/expand hooks
# ---------------------------------------------------------------------------
SERVER_MEMBERS = """

    // recurrent models need shrink/expand of recurrent state around prompt cache operations
    bool is_recurrent_model = false;
    // number of parallel sequences (needed to restore recurrent state after shrink)
    int  n_parallel_user = 0;"""

SERVER_SHRINK_EXPAND = """

    // Shrink recurrent state to 1 cell before saving/loading prompt cache.
    // This frees GPU memory for the cache and ensures a clean recurrent state
    // that matches the loaded KV cache. The subsequent expand restores capacity.
    bool recurrent_shrink_for_prompt_cache() {
        if (!is_recurrent_model) {
            return true;
        }

        if (llama_context_recurrent_shrink(ctx_tgt, 1)) {
            SRV_INF("%s", "shrunk recurrent state to 1 cell for prompt cache\\n");
            return true;
        }

        SRV_ERR("failed to shrink recurrent state (%s)\\n", "prompt cache");
        return false;
    }

    // Expand recurrent state back after prompt cache save/load completes.
    void recurrent_expand_after_prompt_cache() {
        if (!is_recurrent_model) {
            return;
        }

        // Expand to n_parallel_user cells (the original allocation from model init).
        // Context checkpoints will be re-created after this, referencing the new cells.
        if (llama_context_recurrent_expand(ctx_tgt, n_parallel_user)) {
            SRV_INF("expanded recurrent state to %d cells after prompt cache\\n", n_parallel_user);
            return;
        }

        SRV_ERR("failed to expand recurrent state (%s)\\n", "prompt cache");
    }"""

patch_file("tools/server/server-context.cpp", [
    # Add member variables after slot_prompt_similarity
    (
        "float slot_prompt_similarity = 0.0f;",
        None,
        SERVER_MEMBERS,
    ),
    # Initialize is_recurrent_model and n_parallel_user after vocab assignment
    (
        "vocab = llama_model_get_vocab(model_tgt);",
        None,
        """

        is_recurrent_model = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
        n_parallel_user = params_base.n_parallel;""",
    ),
])

# Add shrink/expand methods BEFORE get_available_slot.  Insert before the
# function declaration line, not after it (which would put them inside the body).
path = "tools/server/server-context.cpp"
with open(path, "r") as f:
    src = f.read()

get_slot_anchor = "    server_slot * get_available_slot(const server_task & task) {"
if get_slot_anchor not in src:
    print("PATCH 2.6a FAILED: could not find get_available_slot", file=sys.stderr)
    sys.exit(1)

src = src.replace(get_slot_anchor, SERVER_SHRINK_EXPAND + "\n\n" + get_slot_anchor, 1)
with open(path, "w") as f:
    f.write(src)

# 6b. Wire shrink/expand into the prompt cache update flow
path = "tools/server/server-context.cpp"
with open(path, "r") as f:
    src = f.read()

# Force prompt cache update for recurrent models
old_pattern = """        if (ret) {
            update_cache = update_cache && prompt_cache;"""
new_pattern = """        if (ret) {
            // Force prompt cache update for recurrent models to shrink/restore
            // the recurrent state and avoid forced re-processing (issue #22746).
            if (is_recurrent_model && prompt_cache) {
                update_cache = true;
            }

            update_cache = update_cache && prompt_cache;"""
if old_pattern in src:
    src = src.replace(old_pattern, new_pattern, 1)
else:
    print("PATCH 2.6b: Could not find prompt cache update pattern - recurrent model cache force may not apply", file=sys.stderr)

# Wrap prompt cache update with shrink
old_pattern2 = """            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\\n");"""
new_pattern2 = """            if (update_cache) {
                recurrent_shrink_for_prompt_cache();
                SRV_TRC("%s", "updating prompt cache\\n");"""
if old_pattern2 in src:
    src = src.replace(old_pattern2, new_pattern2, 1)
else:
    print("PATCH 2.6b: Could not find shrink wrap pattern", file=sys.stderr)

# Expand after prompt_cache->update()
old_pattern3 = """                prompt_cache->update();

                SRV_TRC("prompt cache update took"""
new_pattern3 = """                prompt_cache->update();
                recurrent_expand_after_prompt_cache();
                SRV_TRC("prompt cache update took"""
if old_pattern3 in src:
    src = src.replace(old_pattern3, new_pattern3, 1)
else:
    print("PATCH 2.6b: Could not find expand wrap pattern", file=sys.stderr)

with open(path, "w") as f:
    f.write(src)

print("PATCH 2 applied: recurrent shrink/expand API + server integration")
