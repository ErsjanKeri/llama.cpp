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

// --- Runtime ---
//
// Phase1: submit+wait for up+gate, submit down async (same epoch).
// Phase2 wait: drain the down completions.
//
// Initial implementation delegates to the existing 2-phase API in
// llama-io-uring-buf. Subsequent commits replace the internals with
// per-expert async completion polling (all 12 reads submitted up-front,
// compute advances per expert as its 3 weights arrive).

int llama_moe_pipeline_phase1_load(
        struct llama_uring_expert_buf * ebuf,
        int                             layer,
        const int32_t                 * expert_ids,
        int                             n_ids);

int llama_moe_pipeline_phase2_wait(
        struct llama_uring_expert_buf * ebuf);

// --- Fused MoE FFN compute ---
//
// Single-token decode fused path. Submits all 12 io_uring reads, waits for
// completion, then runs per-expert compute:
//   up_e = W_up[e] · x      (+ bias)
//   gate_e = W_gate[e] · x  (+ bias)
//   act_e = swiglu_oai(gate_e, up_e, alpha, limit)
//   down_e = W_down[e] · act_e  (+ bias)
//   y += router_weights[e] * down_e
//
// Multi-threaded: matmul/swiglu/accumulate work is row-split across ith/nth;
// io_uring load and quantization are done by thread 0 only. Synchronisation
// uses an internal sense-reversing barrier in the shared state.

// Opaque per-layer shared state. Allocated once per layer at graph-build time
// (single-threaded), reused across decode invocations. Contains scratch buffers
// and barrier atomics used by the multi-threaded compute function.
struct moe_pipeline_shared;

struct moe_pipeline_shared * moe_pipeline_shared_alloc(int n_embd, int ffn_dim, int n_expert_used);
void moe_pipeline_shared_free(struct moe_pipeline_shared * s);

struct llama_moe_fused_args {
    // Output / inputs (host buffers, already materialized)
    float        * dst_data;              // [n_embd, n_tokens]
    const float  * x_data;                // [n_embd, n_tokens]
    const int32_t* selected_experts;      // [n_expert_used, n_tokens]
    const float  * router_weights;        // [1, n_expert_used, n_tokens]

    // Expert buffer manager for io_uring loads
    struct llama_uring_expert_buf * ebuf;
    int layer;

    // Dimensions
    int n_expert_used;
    int n_tokens;
    int n_embd;
    int ffn_dim;
    int n_expert;   // total experts per layer (for bias indexing)

    // Expert weight types (per projection) — drives vec_dot selection
    int W_up_type;     // enum ggml_type
    int W_gate_type;
    int W_down_type;

    // Swiglu-OAI parameters (set when activation is SWIGLU_OAI_MOE)
    float swiglu_alpha;
    float swiglu_limit;
    int   has_swiglu_oai;  // 1 for GPT-OSS, 0 otherwise

    // Biases (nullable). Shape: [ffn_dim, n_expert] for up/gate; [n_embd, n_expert] for down.
    const void * up_bias_data;
    const void * gate_bias_data;
    const void * down_bias_data;
    int up_bias_type;
    int gate_bias_type;
    int down_bias_type;

    // Threading (passed per-callback invocation).
    int ith;
    int nth;
    struct moe_pipeline_shared * shared;  // scratch buffers + barrier; owned by userdata
};

int llama_moe_pipeline_compute_fused(const struct llama_moe_fused_args * args);

#ifdef __cplusplus
}
#endif

#endif // __linux__
