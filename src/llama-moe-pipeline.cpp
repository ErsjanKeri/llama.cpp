// BSC thesis: per-expert async pipelining for MoE FFN layers.
// See llama-moe-pipeline.h for the design.
//
// Stage: scaffolding only. The pipeline struct and init/free exist so the
// module compiles and links into the llama library; subsequent commits add
// the fused-op kernel, graph integration, and io_uring completion polling.

#ifdef __linux__

#include "llama-moe-pipeline.h"
#include "llama-io-uring-buf.h"

#include <stdlib.h>

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

#endif // __linux__
