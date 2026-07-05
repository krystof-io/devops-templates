#!/usr/bin/env python3
"""
PATCH 1: Fix checkpoint search for hybrid/recurrent models.

Without this, the checkpoint search always returns no valid checkpoint for
Qwen3.6's hybrid GDN+attention architecture, forcing full prompt re-prefill
on every turn (llama.cpp #22384, #20225, #24055).

Applies to: tools/server/server-context.cpp
"""
import sys

path = "tools/server/server-context.cpp"
with open(path, "r") as f:
    src = f.read()

old = "return cur.pos_min < pos_min_thold || cur.pos_min == 0;"
new = (
    "if (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) {\n"
    "                                                return cur.pos_max <= pos_next;\n"
    "                                            }\n"
    "                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;"
)

count = src.count(old)
if count != 1:
    print(f"PATCH 1 FAILED: expected 1 occurrence of target string, found {count}", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("PATCH 1 applied: hybrid/recurrent checkpoint search fix")
