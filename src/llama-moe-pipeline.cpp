// BSC thesis: per-expert async pipelining for MoE FFN layers.
// See llama-moe-pipeline.h for the design.
//
// Stage: scaffolding only. The pipeline struct and init/free exist so the
// module compiles and links into the llama library; subsequent commits add
// the fused-op kernel, graph integration, and io_uring completion polling.

#ifdef __linux__

#include "llama-moe-pipeline.h"
#include "llama-io-uring-buf.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <atomic>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define MOE_CPU_RELAX() _mm_pause()
#else
#define MOE_CPU_RELAX() ((void) 0)
#endif

// Per-layer shared state: scratch buffers for intermediate vectors and the
// sense-reversing barrier used to synchronise stages across threads inside
// one custom-op invocation. Allocated once per layer at graph-build time;
// reused across every decode pass for that layer.
struct moe_pipeline_shared {
    // Intermediate vectors (F32).
    float * up_out;       // [ffn_dim]
    float * gate_out;     // [ffn_dim]
    float * act;          // [ffn_dim]
    float * down_out;     // [n_embd]

    // Quantised activation buffers (Q8_0 layout — large enough for any model
    // we care about since Q8_0 is the common vec_dot_type for F16/MXFP4/Q4_*).
    char  * x_q;           // len = ggml_row_size(Q8_0, n_embd)
    // Per-expert quantised swiglu activation. Stored because up+gate+swiglu
    // for all experts runs BEFORE any down matmul so that phase2 io_uring
    // reads can overlap with the up+gate+swiglu compute — this reclaims the
    // I/O-compute overlap that the existing --uring-overlap path gets.
    char  * act_q_per_expert;  // len = n_expert_used * ggml_row_size(Q8_0, ffn_dim)
    size_t  act_q_row_bytes;   // ggml_row_size(Q8_0, ffn_dim) — stride within act_q_per_expert
    int     n_expert_used;

    // Sense-reversing barrier (atomic spin, no syscall).
    //
    // Each barrier call:
    //   1. Each thread reads `barrier_gen` (relaxed) — its "ticket" for this barrier.
    //   2. Each thread fetch_adds `barrier_count` (seq_cst). The thread that sees
    //      the old value == nth-1 is the last arriver.
    //   3. Last arriver: reset count to 0 (relaxed; the fetch_add seq_cst forms
    //      a fence) and increment `barrier_gen` (seq_cst — release for waiters).
    //   4. Other threads: spin on `barrier_gen.load(acquire)` until it differs
    //      from their ticket.
    //
    // Generation counter monotonically increases over the program lifetime;
    // absolute value is irrelevant, only the change-event matters. ~32-bit
    // counter is safe for any plausible run length (INT_MAX = 2.1B).
    //
    // Cache-line aligned to avoid false sharing between threads contending on
    // count (every barrier writes it) and threads spinning on gen.
    alignas(64) std::atomic<int> barrier_count;
    alignas(64) std::atomic<int> barrier_gen;

    // Captured dimensions for sanity assertions.
    int n_embd;
    int ffn_dim;
};

struct moe_pipeline_shared * moe_pipeline_shared_alloc(int n_embd, int ffn_dim, int n_expert_used) {
    auto * s = new moe_pipeline_shared();
    s->up_out   = (float *) malloc((size_t) ffn_dim * sizeof(float));
    s->gate_out = (float *) malloc((size_t) ffn_dim * sizeof(float));
    s->act      = (float *) malloc((size_t) ffn_dim * sizeof(float));
    s->down_out = (float *) malloc((size_t) n_embd  * sizeof(float));
    s->x_q   = (char *) malloc(ggml_row_size(GGML_TYPE_Q8_0, n_embd));
    s->act_q_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, ffn_dim);
    s->act_q_per_expert = (char *) malloc((size_t) n_expert_used * s->act_q_row_bytes);
    s->n_expert_used = n_expert_used;
    s->barrier_count.store(0, std::memory_order_relaxed);
    s->barrier_gen.store(0, std::memory_order_relaxed);
    s->n_embd = n_embd;
    s->ffn_dim = ffn_dim;
    return s;
}

void moe_pipeline_shared_free(struct moe_pipeline_shared * s) {
    if (!s) return;
    free(s->up_out);
    free(s->gate_out);
    free(s->act);
    free(s->down_out);
    free(s->x_q);
    free(s->act_q_per_expert);
    delete s;
}

// Atomic sense-reversing barrier. Pure userspace, no syscall.
//
// Correctness: all `nth` threads must call this. Last arriver to fetch_add
// the count to nth-1 (returned old value) resets count and bumps gen, which
// releases the spinners. seq_cst on count.fetch_add and gen.fetch_add forms
// a total order; combined with release-acquire on gen, all writes by any
// thread before its fetch_add are visible to all threads after they exit.
//
// Caveat: spins instead of sleeping. Costs CPU during long blocks (e.g. the
// phase2_wait barrier where thread 0 may wait ms on disk). On dedicated CPU
// cores that's pure energy cost — wall time is unaffected because the
// spinning threads have no other work to do. ggml_barrier (used by the
// overlap path) does the same thing, so this is the apples-to-apples choice.
static inline void moe_barrier(moe_pipeline_shared * s, int nth) {
    if (nth <= 1) return;
    const int gen = s->barrier_gen.load(std::memory_order_relaxed);
    if (s->barrier_count.fetch_add(1, std::memory_order_seq_cst) == nth - 1) {
        // Last arriver: reset for next barrier, then advance gen to release.
        // The seq_cst fetch_add on gen acts as a release of all writes this
        // thread made before this point.
        s->barrier_count.store(0, std::memory_order_relaxed);
        s->barrier_gen.fetch_add(1, std::memory_order_seq_cst);
    } else {
        while (s->barrier_gen.load(std::memory_order_acquire) == gen) {
            MOE_CPU_RELAX();
        }
    }
}

struct llama_moe_pipeline {
    struct llama_uring_expert_buf * ebuf;
    // Per-layer, per-expert state will be added when the fused op lands.
};

struct llama_moe_pipeline * llama_moe_pipeline_init(
        struct llama_uring_expert_buf * ebuf) {
    if (ebuf == nullptr) {
        return nullptr;
    }
    auto * p = (llama_moe_pipeline *) calloc(1, sizeof(llama_moe_pipeline));
    if (p == nullptr) {
        return nullptr;
    }
    p->ebuf = ebuf;
    return p;
}

void llama_moe_pipeline_free(struct llama_moe_pipeline * pipeline) {
    if (pipeline == nullptr) {
        return;
    }
    free(pipeline);
}

// --- Runtime wrappers ---
//
// Initial implementation: delegate to the existing 2-phase API. This makes the
// --uring-pipeline path byte-identical to --uring-overlap but routed through
// this module, so subsequent commits can replace the internals with real
// per-expert async pipelining without touching the graph builder.

int llama_moe_pipeline_phase1_load(
        struct llama_uring_expert_buf * ebuf,
        int                             layer,
        const int32_t                 * expert_ids,
        int                             n_ids) {
    int ret = llama_uring_expert_buf_load_phase1(ebuf, layer, expert_ids, n_ids);
    if (ret < 0) {
        return ret;
    }
    return llama_uring_expert_buf_load_phase2_submit(ebuf, layer, expert_ids, n_ids);
}

int llama_moe_pipeline_phase2_wait(struct llama_uring_expert_buf * ebuf) {
    return llama_uring_expert_buf_load_phase2_wait(ebuf);
}

// --- Fused MoE FFN compute (single-threaded, sequential per expert) ---
//
// Goal of this first pass: bit-exact (or near bit-exact) match with the
// existing graph path, *without* pipelining yet. Validates that we can
// compute the full FFN block inside one custom op using ggml's own vec_dot.
// Multi-thread and real async pipelining come in subsequent commits.

// swiglu_oai per-element — exactly matches ggml_compute_forward_swiglu_oai_f32.
static inline void moe_swiglu_oai_range(float * out, const float * gate, const float * up,
                                         int k_start, int k_end, float alpha, float limit) {
    for (int k = k_start; k < k_end; k++) {
        const float x = std::min(gate[k], limit);
        const float y = std::clamp(up[k], -limit, limit);
        const float out_glu = x / (1.f + expf(alpha * (-x)));
        out[k] = out_glu * (y + 1.f);
    }
}

// Matmul for a single token, producing output rows [i_start, i_end).
// out[i] = Σ_k W[k, i] * x_q[k], for i in the thread's slice.
// W_data + i*K_bytes = start of W-row i (ggml storage: nb[1] = K_bytes per row).
static inline void moe_matvec_range(float * out, const void * W_data, const void * x_q,
                                     int K, int i_start, int i_end, size_t K_bytes,
                                     ggml_vec_dot_t vec_dot) {
    for (int i = i_start; i < i_end; i++) {
        const char * W_row = (const char *) W_data + (size_t) i * K_bytes;
        vec_dot(K, &out[i], 0, W_row, 0, x_q, 0, 1);
    }
}

// Compute the [start, end) row range this thread owns when splitting N across nth threads.
static inline void moe_row_range(int N, int ith, int nth, int * start, int * end) {
    const int rows_per_thread = (N + nth - 1) / nth;
    *start = std::min(ith * rows_per_thread, N);
    *end   = std::min(*start + rows_per_thread, N);
}

int llama_moe_pipeline_compute_fused(const struct llama_moe_fused_args * args) {
    if (args == nullptr || args->ebuf == nullptr || args->shared == nullptr) {
        return -1;
    }
    if (args->n_tokens != 1) {
        if (args->ith == 0) {
            fprintf(stderr, "moe-pipeline: n_tokens=%d, only single-token decode supported\n", args->n_tokens);
        }
        return -1;
    }
    if (!args->has_swiglu_oai) {
        if (args->ith == 0) {
            fprintf(stderr, "moe-pipeline: only SWIGLU_OAI activation supported in this first pass\n");
        }
        return -1;
    }

    const int n_embd        = args->n_embd;
    const int ffn_dim       = args->ffn_dim;
    const int n_expert_used = args->n_expert_used;
    const int ith           = args->ith;
    const int nth           = args->nth;
    moe_pipeline_shared * sh = args->shared;

    // Sanity: buffer sizes were fixed at alloc time.
    if (sh->n_embd != n_embd || sh->ffn_dim != ffn_dim) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline: shared dims mismatch (alloc %d/%d, op %d/%d)\n",
                    sh->n_embd, sh->ffn_dim, n_embd, ffn_dim);
        }
        return -1;
    }

    const ggml_type_traits_cpu * up_traits   = ggml_get_type_traits_cpu((ggml_type) args->W_up_type);
    const ggml_type_traits_cpu * gate_traits = ggml_get_type_traits_cpu((ggml_type) args->W_gate_type);
    const ggml_type_traits_cpu * down_traits = ggml_get_type_traits_cpu((ggml_type) args->W_down_type);

    const ggml_type_traits_cpu * up_qtraits   = ggml_get_type_traits_cpu(up_traits->vec_dot_type);
    const ggml_type_traits_cpu * down_qtraits = ggml_get_type_traits_cpu(down_traits->vec_dot_type);

    // Assumption for the shared x_q buffer: up and gate use the same vec_dot_type
    // (true for GPT-OSS, where both are MXFP4 → Q8_0). If a future model breaks
    // this, add a separate x_gate_q buffer in shared state.
    if (up_traits->vec_dot_type != gate_traits->vec_dot_type) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline: up/gate vec_dot_type mismatch not supported\n");
        }
        return -1;
    }

    const size_t up_K_bytes   = ggml_row_size((ggml_type) args->W_up_type, n_embd);
    const size_t gate_K_bytes = ggml_row_size((ggml_type) args->W_gate_type, n_embd);
    const size_t down_K_bytes = ggml_row_size((ggml_type) args->W_down_type, ffn_dim);

    // =============================================================================
    // Stage 1: kick off all I/O, quantise x, zero the output.
    //   - phase1: synchronous load of up+gate (8 slots)
    //   - phase2_submit: async load of down (4 slots) — keeps running during
    //     the up+gate+swiglu compute that follows.
    // =============================================================================
    if (ith == 0) {
        int32_t expert_ids[LLAMA_URING_MAX_EXPERTS];
        for (int e = 0; e < n_expert_used; e++) {
            expert_ids[e] = args->selected_experts[e];
        }
        int r1 = llama_uring_expert_buf_load_phase1(args->ebuf, args->layer, expert_ids, n_expert_used);
        int r2 = (r1 == 0) ? llama_uring_expert_buf_load_phase2_submit(args->ebuf, args->layer, expert_ids, n_expert_used) : r1;
        if (r2 < 0) {
            fprintf(stderr, "moe-pipeline: phase1/phase2_submit failed for layer %d\n", args->layer);
        }
        up_qtraits->from_float(args->x_data, sh->x_q, n_embd);
        memset(args->dst_data, 0, sizeof(float) * n_embd);
    }
    moe_barrier(sh, nth);

    // up and gate slot pointers are valid after phase1 (sync). down slots will
    // be valid only after phase2_wait — read them after the stage-3 barrier.
    const void * const * gate_slots = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 1);
    const void * const * up_slots   = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 2);

    // =============================================================================
    // Stage 2: up + gate + swiglu for every expert, using thread-local row slices.
    //   The down-projection io_uring reads submitted in stage 1 continue in
    //   parallel with this compute. Each expert's swiglu result is quantised to
    //   Q8_0 into a per-expert slot in `act_q_per_expert` so stage 4 can replay
    //   them in any order.
    //
    //   All operations within one expert's iteration read/write only the
    //   calling thread's [i0, i1) slice, so NO barrier is needed between
    //   up/gate/swiglu. Only one barrier per expert is required before thread 0
    //   quantises `act` (which reads the full range).
    // =============================================================================
    for (int e = 0; e < n_expert_used; e++) {
        const int32_t eid = args->selected_experts[e];
        int i0, i1;
        moe_row_range(ffn_dim, ith, nth, &i0, &i1);

        // up projection
        moe_matvec_range(sh->up_out, up_slots[e], sh->x_q,
                         n_embd, i0, i1, up_K_bytes, up_traits->vec_dot);
        if (args->up_bias_data) {
            const float * up_b = (const float *) args->up_bias_data + (size_t) eid * ffn_dim;
            for (int k = i0; k < i1; k++) sh->up_out[k] += up_b[k];
        }

        // gate projection
        moe_matvec_range(sh->gate_out, gate_slots[e], sh->x_q,
                         n_embd, i0, i1, gate_K_bytes, gate_traits->vec_dot);
        if (args->gate_bias_data) {
            const float * gate_b = (const float *) args->gate_bias_data + (size_t) eid * ffn_dim;
            for (int k = i0; k < i1; k++) sh->gate_out[k] += gate_b[k];
        }

        // swiglu_oai (same i0..i1 slice — up_out[k] and gate_out[k] were both
        // written above by this same thread, so no cross-thread read).
        moe_swiglu_oai_range(sh->act, sh->gate_out, sh->up_out, i0, i1,
                             args->swiglu_alpha, args->swiglu_limit);

        // Sync before thread 0 reads the full `act` for quantisation.
        moe_barrier(sh, nth);

        if (ith == 0) {
            char * act_q_e = sh->act_q_per_expert + (size_t) e * sh->act_q_row_bytes;
            down_qtraits->from_float(sh->act, act_q_e, ffn_dim);
        }
        // No barrier here — the next expert's up/gate/swiglu reads x_q and
        // up_slots[e+1] only, not act or act_q. Thread 0's quantise will be
        // visible via the stage-3 barrier before stage 4 reads act_q_per_expert.
    }

    // =============================================================================
    // Stage 3: wait for phase2 (down-projection) reads to complete.
    //   By now stage 2's ~(n_expert_used × (2 matmuls + swiglu + quantise)) of
    //   compute has elapsed, during which the NVMe completed the 4 down reads.
    //   phase2_wait is the final sync; on any remaining tail it blocks here.
    // =============================================================================
    if (ith == 0) {
        int r = llama_uring_expert_buf_load_phase2_wait(args->ebuf);
        if (r < 0) {
            fprintf(stderr, "moe-pipeline: phase2_wait failed for layer %d\n", args->layer);
        }
    }
    moe_barrier(sh, nth);

    const void * const * down_slots = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 0);

    // =============================================================================
    // Stage 4: down matmul + bias + router-weighted accumulation for every expert.
    //   Everything here is thread-local — each thread owns rows [i0, i1) of the
    //   n_embd output. dst_data is written only by the thread that owns that row,
    //   both for the zero-init in stage 1 (thread 0 full) and for the per-expert
    //   accumulate here (own slice). No barriers needed within the loop.
    //
    //   The implicit ggml_barrier after the custom op returns provides the
    //   cross-thread visibility the next graph node (residual add) needs.
    // =============================================================================
    {
        int i0, i1;
        moe_row_range(n_embd, ith, nth, &i0, &i1);
        for (int e = 0; e < n_expert_used; e++) {
            const int32_t eid = args->selected_experts[e];
            const char * act_q_e = sh->act_q_per_expert + (size_t) e * sh->act_q_row_bytes;

            moe_matvec_range(sh->down_out, down_slots[e], act_q_e,
                             ffn_dim, i0, i1, down_K_bytes, down_traits->vec_dot);
            if (args->down_bias_data) {
                const float * down_b = (const float *) args->down_bias_data + (size_t) eid * n_embd;
                for (int k = i0; k < i1; k++) sh->down_out[k] += down_b[k];
            }
            const float w = args->router_weights[e];
            for (int k = i0; k < i1; k++) args->dst_data[k] += w * sh->down_out[k];
        }
    }

    return 0;
}

#endif // __linux__
