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

#include <pthread.h>

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
    char  * x_q;          // len = ggml_row_size(Q8_0, n_embd)
    char  * act_q;        // len = ggml_row_size(Q8_0, ffn_dim)

    // pthread barrier — reusable across every stage and every op invocation
    // (pthread_barrier_wait handles the reset automatically). Initialised with
    // the thread count at first use; if the thread count changes later the
    // barrier is reinitialised.
    pthread_barrier_t barrier;
    int barrier_nth;  // thread count the barrier was initialised for; 0 = uninit
    pthread_mutex_t init_mutex;

    // Captured dimensions for sanity assertions.
    int n_embd;
    int ffn_dim;
};

struct moe_pipeline_shared * moe_pipeline_shared_alloc(int n_embd, int ffn_dim) {
    auto * s = new moe_pipeline_shared();
    s->up_out   = (float *) malloc((size_t) ffn_dim * sizeof(float));
    s->gate_out = (float *) malloc((size_t) ffn_dim * sizeof(float));
    s->act      = (float *) malloc((size_t) ffn_dim * sizeof(float));
    s->down_out = (float *) malloc((size_t) n_embd  * sizeof(float));
    // Q8_0 row size = (n/32) * 34. Use the exact ggml helper so the layout
    // matches what vec_dot_*_q8_0 expects.
    s->x_q   = (char *) malloc(ggml_row_size(GGML_TYPE_Q8_0, n_embd));
    s->act_q = (char *) malloc(ggml_row_size(GGML_TYPE_Q8_0, ffn_dim));
    s->barrier_nth = 0;
    pthread_mutex_init(&s->init_mutex, nullptr);
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
    free(s->act_q);
    if (s->barrier_nth > 0) {
        pthread_barrier_destroy(&s->barrier);
    }
    pthread_mutex_destroy(&s->init_mutex);
    delete s;
}

// pthread-based barrier. All `nth` threads must call this before any of them
// proceeds. Init path is mutex-guarded; after first initialisation with a
// given `nth`, the fast path is a single pthread_barrier_wait.
static inline void moe_barrier(moe_pipeline_shared * s, int nth) {
    if (nth <= 1) return;
    pthread_mutex_lock(&s->init_mutex);
    if (s->barrier_nth != nth) {
        if (s->barrier_nth > 0) {
            pthread_barrier_destroy(&s->barrier);
        }
        pthread_barrier_init(&s->barrier, nullptr, nth);
        s->barrier_nth = nth;
    }
    pthread_mutex_unlock(&s->init_mutex);
    pthread_barrier_wait(&s->barrier);
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

    const size_t up_K_bytes   = ggml_row_size((ggml_type) args->W_up_type, n_embd);
    const size_t gate_K_bytes = ggml_row_size((ggml_type) args->W_gate_type, n_embd);
    const size_t down_K_bytes = ggml_row_size((ggml_type) args->W_down_type, ffn_dim);

    // --- Stage 1: thread 0 submits io_uring loads and quantises x. ---
    // For now, assume up and gate share the same vec_dot_type (true for GPT-OSS).
    // If they differ, we'd need a second x_q buffer — add later if a model needs it.
    if (ith == 0) {
        int32_t expert_ids[LLAMA_URING_MAX_EXPERTS];
        for (int e = 0; e < n_expert_used; e++) {
            expert_ids[e] = args->selected_experts[e];
        }
        int r1 = llama_uring_expert_buf_load_phase1(args->ebuf, args->layer, expert_ids, n_expert_used);
        int r2 = (r1 == 0) ? llama_uring_expert_buf_load_phase2_submit(args->ebuf, args->layer, expert_ids, n_expert_used) : r1;
        int r3 = (r2 == 0) ? llama_uring_expert_buf_load_phase2_wait(args->ebuf) : r2;
        if (r3 < 0) {
            fprintf(stderr, "moe-pipeline: io_uring load failed for layer %d\n", args->layer);
        }
        up_qtraits->from_float(args->x_data, sh->x_q, n_embd);
        memset(args->dst_data, 0, sizeof(float) * n_embd);
    }
    moe_barrier(sh, nth);

    // Slot pointers become valid after phase2_wait completes — read after the
    // barrier so all threads see the same post-load pointers.
    const void * const * down_slots = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 0);
    const void * const * gate_slots = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 1);
    const void * const * up_slots   = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 2);

    // Per-expert loop. Each stage within the expert is row-split across threads
    // and followed by a barrier so dependent stages see a consistent buffer.
    for (int e = 0; e < n_expert_used; e++) {
        const int32_t eid = args->selected_experts[e];

        // up projection (rows split across threads). Bias add runs on the same
        // [i0, i1) range so each thread only touches its own rows — avoids a
        // cross-thread read before the barrier.
        {
            int i0, i1;
            moe_row_range(ffn_dim, ith, nth, &i0, &i1);
            moe_matvec_range(sh->up_out, up_slots[e], sh->x_q,
                             n_embd, i0, i1, up_K_bytes, up_traits->vec_dot);
            if (args->up_bias_data) {
                const float * up_b = (const float *) args->up_bias_data + (size_t) eid * ffn_dim;
                for (int k = i0; k < i1; k++) sh->up_out[k] += up_b[k];
            }
        }
        moe_barrier(sh, nth);

        // gate projection (same thread-local slice pattern)
        {
            int i0, i1;
            moe_row_range(ffn_dim, ith, nth, &i0, &i1);
            moe_matvec_range(sh->gate_out, gate_slots[e], sh->x_q,
                             n_embd, i0, i1, gate_K_bytes, gate_traits->vec_dot);
            if (args->gate_bias_data) {
                const float * gate_b = (const float *) args->gate_bias_data + (size_t) eid * ffn_dim;
                for (int k = i0; k < i1; k++) sh->gate_out[k] += gate_b[k];
            }
        }
        moe_barrier(sh, nth);

        // swiglu_oai — split elements across threads (reads up_out, gate_out).
        {
            int k0, k1;
            moe_row_range(ffn_dim, ith, nth, &k0, &k1);
            moe_swiglu_oai_range(sh->act, sh->gate_out, sh->up_out, k0, k1,
                                 args->swiglu_alpha, args->swiglu_limit);
        }
        moe_barrier(sh, nth);

        // Quantise activation → down's vec_dot_type (thread 0 only; from_float
        // APIs typically aren't parallel-safe on a single buffer).
        if (ith == 0) {
            down_qtraits->from_float(sh->act, sh->act_q, ffn_dim);
        }
        moe_barrier(sh, nth);

        // down projection + bias + weighted accumulate — same thread-local slice.
        {
            int i0, i1;
            moe_row_range(n_embd, ith, nth, &i0, &i1);
            moe_matvec_range(sh->down_out, down_slots[e], sh->act_q,
                             ffn_dim, i0, i1, down_K_bytes, down_traits->vec_dot);
            if (args->down_bias_data) {
                const float * down_b = (const float *) args->down_bias_data + (size_t) eid * n_embd;
                for (int k = i0; k < i1; k++) sh->down_out[k] += down_b[k];
            }
            const float w = args->router_weights[e];
            for (int k = i0; k < i1; k++) args->dst_data[k] += w * sh->down_out[k];
        }
        moe_barrier(sh, nth);
    }

    return 0;
}

#endif // __linux__
