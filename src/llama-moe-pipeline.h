#pragma once

// BSC thesis: per-expert async pipelining for MoE FFN layers.
//
// Replaces the standard graph subgraph (mm_id(up), mm_id(gate), swiglu,
// mm_id(down), accumulate) with a single fused custom ggml op. Inside the op,
// io_uring expert-weight reads are submitted up-front with per-(expert) or
// per-(expert, projection) tags, and compute advances per expert as that
// expert's weights arrive — instead of waiting for the slowest read.
//
// Two variants share this scaffolding:
//   --uring-async-projection-overlap — split-tag pipelining (upgate-pair + down per expert)
//   --uring-async-experts            — per-(expert, projection) tags + eager dispatch
//
// Both target the same token stream as --uring-projection-overlap (seed 42)
// over 2000 tokens. Scope: decode only (n_tokens == 1). Prompt eval continues
// to use the default MoE graph.

#ifdef __linux__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_uring_expert_buf;  // forward decl from llama-io-uring-buf.h

// --- Fused MoE FFN compute ---
//
// Single-token decode fused path. Submits per-expert (or per-(expert, projection))
// tagged io_uring reads at layer entry and runs:
//   up_e   = W_up[e] · x      (+ bias)
//   gate_e = W_gate[e] · x    (+ bias)
//   act_e  = swiglu_oai(gate_e, up_e, alpha, limit)
//   down_e = W_down[e] · act_e (+ bias)
//   y     += router_weights[e] * down_e
//
// Multi-threaded: matmul/swiglu/accumulate work is row-split across ith/nth;
// io_uring submit/wait and quantization are done by thread 0 only.
// Synchronisation uses an internal sense-reversing atomic barrier.

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

// async-projection-overlap: split-tag (upgate-pair + down per expert) +
// first-ready dynamic dispatch. Submits 4 upgate-pairs (tags 0,2,4,6) + 4
// down-singles (tags 1,3,5,7) at layer entry, then dispatches whichever
// expert's upgate pair arrives first. Down matmul runs after swiglu; its I/O
// (tag 2e+1) is usually already complete by then.
//
// Activated by --uring-async-projection-overlap.
int llama_moe_pipeline_compute_async_projection_overlap(const struct llama_moe_fused_args * args);

// async-experts: per-(expert, projection) tags + eager dispatch. Submits 12
// individual io_uring reads (4 experts x 3 projections: up, gate, down) with
// distinct tags 0-11, and dispatches each matmul as soon as its weight
// arrives, instead of waiting for the upgate-pair as async-projection-overlap
// does.
//
// Compute order across experts is event-driven (whichever expert's matmul
// becomes eligible first runs first). Output is stored indexed by expert-id
// and the final reduction runs in fixed expert-id order so the result is
// bit-exact vs async-projection-overlap (and vs --uring-projection-overlap,
// since async-projection-overlap is bit-exact vs that path).
//
// Activated by --uring-async-experts.
int llama_moe_pipeline_compute_async_experts(const struct llama_moe_fused_args * args);

// --- Profiling counters (BSC May 1 2026) ---
//
// Process-global accumulators across all moe_pipeline_shared instances and
// all decode steps. Used to attribute the async-projection-overlap vs
// async-experts wall-clock gap.
//
//   barrier_count    : number of barrier crossings (one increment per crossing,
//                      written by the unique last-arriver thread).
//   barrier_wait_ns  : sum of per-thread time spent spinning in moe_barrier
//                      (each waiter adds its own elapsed wait on exit).
//
// Average wait per barrier per waiter = barrier_wait_ns / (barrier_count * (nth-1))
// when all barriers use the same nth.
uint64_t llama_moe_pipeline_get_barrier_count(void);
uint64_t llama_moe_pipeline_get_barrier_wait_ns(void);
void     llama_moe_pipeline_reset_barrier_counters(void);

// async-experts picker attribution (May 1 2026): partition thread 0's
// pre-barrier time into pure compute (state machine + bookkeeping) and
// blocking I/O (wait_any_upgate_ready). Only meaningful under
// --uring-async-experts; remain 0 under --uring-async-projection-overlap.
uint64_t llama_moe_pipeline_get_ae_t0_pick_compute_ns(void);
uint64_t llama_moe_pipeline_get_ae_t0_pick_io_wait_ns(void);

#ifdef __cplusplus
}
#endif

#endif // __linux__
