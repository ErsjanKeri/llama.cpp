#pragma once

// BSC thesis: per-expert async pipelining for MoE FFN layers.
//
// Replaces the current graph subgraph (mm_id(up) → mm_id(gate) → swiglu →
// mm_id(down) → accumulate) with a single fused custom ggml op. Inside the op,
// all 12 io_uring expert-weight reads (4 experts × 3 projections) are submitted
// up-front; compute advances per expert as that expert's weights arrive, rather
// than waiting for the slowest read. Uses io_uring as an async primitive
// instead of the current submit+wait pattern.
//
// Correctness target: produce the same token stream (seed 42) as the existing
// --uring-overlap path over 2000 tokens.
//
// Activated by --uring-pipeline (mutually exclusive with --uring-overlap).
//
// Scope: decode only (n_tokens == 1). Prompt eval continues to use the
// existing MoE graph.

#ifdef __linux__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_uring_expert_buf;  // forward decl from llama-io-uring-buf.h

// Per-expert progress tracking inside the fused op.
enum llama_moe_pipeline_stage {
    LLAMA_MOE_STAGE_WAITING_WEIGHTS = 0,  // nothing done yet
    LLAMA_MOE_STAGE_UP_DONE         = 1,  // mm(up, e) computed
    LLAMA_MOE_STAGE_GATE_DONE       = 2,  // mm(gate, e) computed
    LLAMA_MOE_STAGE_SWIGLU_DONE     = 3,  // swiglu(up, gate) computed
    LLAMA_MOE_STAGE_DONE            = 4,  // mm(down, e) computed, accumulated into y
};

// Opaque handle.
struct llama_moe_pipeline;

// --- Lifecycle ---

struct llama_moe_pipeline * llama_moe_pipeline_init(
        struct llama_uring_expert_buf * ebuf);

void llama_moe_pipeline_free(struct llama_moe_pipeline * pipeline);

#ifdef __cplusplus
}
#endif

#endif // __linux__
