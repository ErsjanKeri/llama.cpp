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
#include <vector>

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

static void moe_swiglu_oai(float * out, const float * gate, const float * up,
                            int nc, float alpha, float limit) {
    for (int k = 0; k < nc; k++) {
        const float x = std::min(gate[k], limit);
        const float y = std::clamp(up[k], -limit, limit);
        const float out_glu = x / (1.f + expf(alpha * (-x)));
        out[k] = out_glu * (y + 1.f);
    }
}

// Compute one matmul for a single token: out[i] = Σ_k W[k, i] * x_q[k]
// W_data points to the expert's weight slice (K × N, ggml storage: nb[0]=elem, nb[1]=K*elem).
// x_q has already been converted to vec_dot_type.
static void moe_matvec(float * out, const void * W_data, const void * x_q,
                        int K, int N, size_t K_bytes,
                        ggml_vec_dot_t vec_dot) {
    for (int i = 0; i < N; i++) {
        const char * W_row = (const char *) W_data + (size_t) i * K_bytes;
        vec_dot(K, &out[i], 0, W_row, 0, x_q, 0, 1);
    }
}

// Add a bias row (F32 assumed for now) of length n into out.
static void moe_add_bias_f32(float * out, const float * bias, int n) {
    for (int i = 0; i < n; i++) {
        out[i] += bias[i];
    }
}

int llama_moe_pipeline_compute_fused(const struct llama_moe_fused_args * args) {
    if (args == nullptr || args->ebuf == nullptr) {
        return -1;
    }
    if (args->n_tokens != 1) {
        // Single-token decode only for this first implementation.
        fprintf(stderr, "moe-pipeline: n_tokens=%d, only single-token decode supported\n", args->n_tokens);
        return -1;
    }
    if (!args->has_swiglu_oai) {
        fprintf(stderr, "moe-pipeline: only SWIGLU_OAI activation supported in this first pass\n");
        return -1;
    }

    const int n_embd       = args->n_embd;
    const int ffn_dim      = args->ffn_dim;
    const int n_expert     = args->n_expert;
    const int n_expert_used = args->n_expert_used;

    // Submit all 12 reads (via the existing phase1+phase2 wrappers), wait for all.
    // Sequential semantics for this first pass; async pipelining added later.
    int32_t expert_ids[LLAMA_URING_MAX_EXPERTS];
    for (int e = 0; e < n_expert_used; e++) {
        expert_ids[e] = args->selected_experts[e];  // token 0 only
    }
    int ret = llama_uring_expert_buf_load_phase1(args->ebuf, args->layer, expert_ids, n_expert_used);
    if (ret < 0) {
        return ret;
    }
    ret = llama_uring_expert_buf_load_phase2_submit(args->ebuf, args->layer, expert_ids, n_expert_used);
    if (ret < 0) {
        return ret;
    }
    ret = llama_uring_expert_buf_load_phase2_wait(args->ebuf);
    if (ret < 0) {
        return ret;
    }

    // Slot pointers per projection (proj order: 0=down, 1=gate, 2=up)
    const void * const * down_slots = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 0);
    const void * const * gate_slots = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 1);
    const void * const * up_slots   = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 2);

    // Weight traits (for vec_dot function) and activation-quant traits (for from_float).
    // vec_dot expects the activation already converted to the weight's vec_dot_type;
    // to produce that, we must use the vec_dot_type's from_float, NOT the weight type's.
    const ggml_type_traits_cpu * up_traits   = ggml_get_type_traits_cpu((ggml_type) args->W_up_type);
    const ggml_type_traits_cpu * gate_traits = ggml_get_type_traits_cpu((ggml_type) args->W_gate_type);
    const ggml_type_traits_cpu * down_traits = ggml_get_type_traits_cpu((ggml_type) args->W_down_type);

    const ggml_type_traits_cpu * up_qtraits   = ggml_get_type_traits_cpu(up_traits->vec_dot_type);
    const ggml_type_traits_cpu * gate_qtraits = ggml_get_type_traits_cpu(gate_traits->vec_dot_type);
    const ggml_type_traits_cpu * down_qtraits = ggml_get_type_traits_cpu(down_traits->vec_dot_type);

    const size_t up_q_bytes   = ggml_row_size(up_traits->vec_dot_type, n_embd);
    const size_t gate_q_bytes = ggml_row_size(gate_traits->vec_dot_type, n_embd);
    const size_t down_q_bytes = ggml_row_size(down_traits->vec_dot_type, ffn_dim);

    std::vector<char> x_up_q(up_q_bytes);
    std::vector<char> x_gate_q(up_traits->vec_dot_type == gate_traits->vec_dot_type ? 0 : gate_q_bytes);
    // down's "x" is the swiglu activation (ffn_dim), converted inside the per-expert loop.
    std::vector<char> act_down_q(down_q_bytes);

    up_qtraits->from_float(args->x_data, x_up_q.data(), n_embd);
    const void * x_gate_q_ptr = nullptr;
    if (up_traits->vec_dot_type == gate_traits->vec_dot_type) {
        x_gate_q_ptr = x_up_q.data();
    } else {
        gate_qtraits->from_float(args->x_data, x_gate_q.data(), n_embd);
        x_gate_q_ptr = x_gate_q.data();
    }

    // Stride in bytes to skip one full K-row inside an expert's weight block.
    const size_t up_K_bytes   = ggml_row_size((ggml_type) args->W_up_type, n_embd);
    const size_t gate_K_bytes = ggml_row_size((ggml_type) args->W_gate_type, n_embd);
    const size_t down_K_bytes = ggml_row_size((ggml_type) args->W_down_type, ffn_dim);

    // Zero the output accumulator.
    memset(args->dst_data, 0, sizeof(float) * n_embd);

    // Per-expert buffers.
    std::vector<float> up_out(ffn_dim);
    std::vector<float> gate_out(ffn_dim);
    std::vector<float> act(ffn_dim);
    std::vector<float> down_out(n_embd);

    for (int e = 0; e < n_expert_used; e++) {
        const int32_t eid = args->selected_experts[e];  // original expert id (for biases)

        // up projection
        moe_matvec(up_out.data(),   up_slots[e],   x_up_q.data(),   n_embd, ffn_dim, up_K_bytes,   up_traits->vec_dot);
        if (args->up_bias_data) {
            const float * up_b = (const float *) args->up_bias_data + (size_t) eid * ffn_dim;
            moe_add_bias_f32(up_out.data(), up_b, ffn_dim);
        }

        // gate projection
        moe_matvec(gate_out.data(), gate_slots[e], x_gate_q_ptr,    n_embd, ffn_dim, gate_K_bytes, gate_traits->vec_dot);
        if (args->gate_bias_data) {
            const float * gate_b = (const float *) args->gate_bias_data + (size_t) eid * ffn_dim;
            moe_add_bias_f32(gate_out.data(), gate_b, ffn_dim);
        }

        // swiglu_oai(gate, up, alpha, limit) — matches ggml_compute_forward_swiglu_oai_f32
        moe_swiglu_oai(act.data(), gate_out.data(), up_out.data(), ffn_dim,
                       args->swiglu_alpha, args->swiglu_limit);

        // convert activation → down's vec_dot_type (use vec_dot_type's from_float, not weight type's)
        down_qtraits->from_float(act.data(), act_down_q.data(), ffn_dim);

        // down projection
        moe_matvec(down_out.data(), down_slots[e], act_down_q.data(), ffn_dim, n_embd, down_K_bytes, down_traits->vec_dot);
        if (args->down_bias_data) {
            const float * down_b = (const float *) args->down_bias_data + (size_t) eid * n_embd;
            moe_add_bias_f32(down_out.data(), down_b, n_embd);
        }

        // router-weighted accumulation
        const float w = args->router_weights[e];  // token 0
        for (int i = 0; i < n_embd; i++) {
            args->dst_data[i] += w * down_out[i];
        }

        (void) n_expert;  // silence unused in case bias indexing is skipped
    }

    return 0;
}

#endif // __linux__
