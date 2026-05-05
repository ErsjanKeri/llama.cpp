// BSC thesis: per-expert async pipelining for MoE FFN layers.
// See llama-moe-pipeline.h for the design and the two compute entry points
// (compute_async_projection_overlap and compute_async_experts).

#ifdef __linux__

#include "llama-moe-pipeline.h"
#include "llama-io-uring-buf.h"
#include "llama-io-uring.h"       // LLAMA_IO_URING_MAX_TAGS

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <atomic>
#include <vector>
#include <time.h>

// async-experts picker profiling counters (BSC May 1 2026, gap attribution
// after barrier-count optimization didn't move wall-clock).
//   ae_t0_pick_compute_ns : cumulative thread-0 time in pick logic, EXCLUDING
//                           any blocking wait_any_upgate_ready calls
//   ae_t0_pick_io_wait_ns : cumulative thread-0 time blocked inside
//                           wait_any_upgate_ready (synchronous I/O wait)
// Both are written only by thread 0 across all layer calls, so they could be
// plain uint64 — but we expose them as atomic getters for the JSON dump
// reader to use a stable interface.
static std::atomic<uint64_t> g_ae_t0_pick_compute_ns{0};
static std::atomic<uint64_t> g_ae_t0_pick_io_wait_ns{0};

extern "C" {
uint64_t llama_moe_pipeline_get_ae_t0_pick_compute_ns(void) {
    return g_ae_t0_pick_compute_ns.load(std::memory_order_relaxed);
}
uint64_t llama_moe_pipeline_get_ae_t0_pick_io_wait_ns(void) {
    return g_ae_t0_pick_io_wait_ns.load(std::memory_order_relaxed);
}
}

// async-experts deadlock-debug logging. Off by default. Enable with:
//   BSC_ASYNC_EXPERTS_DEBUG=1 build/bin/llama-completion ...
// Logs every picker decision and every barrier crossing on thread 0 to stderr.
// Logging is read-once at module load to avoid getenv() overhead per call.
static const bool kAsyncExpertsDebug = []() {
    const char * e = getenv("BSC_ASYNC_EXPERTS_DEBUG");
    return e && *e && *e != '0';
}();
static inline int64_t ae_dbg_now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000;
}
#define AE_DBG(fmt, ...) do { if (kAsyncExpertsDebug) fprintf(stderr, "[async-experts t=%lld] " fmt "\n", (long long)ae_dbg_now_us(), ##__VA_ARGS__); } while (0)

// Per-thread location tracker. Each worker writes its current source line and
// outer-iteration counter to its own slot in moe_pipeline_shared. The
// moe_barrier watchdog dumps all entries to identify which thread is stuck
// and where it last made progress. Writes are unsynchronized but safe (each
// thread writes only its own slot; race-on-read is acceptable for diagnostic).
#define AE_MARK(sh, ith, dbg_round) do { \
    if ((sh) && (ith) >= 0 && (ith) < 16) { \
        (sh)->ae_thread_mark[ith] = __LINE__; \
        (sh)->ae_thread_iter[ith] = (dbg_round); \
    } \
} while (0)

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
    // I/O-compute overlap that --uring-projection-overlap gets.
    char  * act_q_per_expert;  // len = n_expert_used * ggml_row_size(Q8_0, ffn_dim)
    size_t  act_q_row_bytes;   // ggml_row_size(Q8_0, ffn_dim) — stride within act_q_per_expert
    int     n_expert_used;

    // Per-expert slot pointers for async pipeline. Thread 0 fills these
    // after each load; other threads read after a barrier.
    const void * e_up[LLAMA_URING_MAX_EXPERTS];
    const void * e_gate[LLAMA_URING_MAX_EXPERTS];
    const void * e_down[LLAMA_URING_MAX_EXPERTS];

    // async-projection-overlap dynamic dispatch: thread 0 writes the order in
    // which experts are dispatched (first-ready first); other threads read
    // after barrier. Index i holds the expert slot (0..n_expert_used-1) to
    // compute at iter i.
    int dispatched_order[LLAMA_URING_MAX_EXPERTS];

    // async-projection-overlap per-expert weighted output buffer:
    // expert_weighted_out[e * n_embd + k] holds router_weights[e] * down_e[k].
    // Populated during the compute loop in dispatch order (iter i writes to
    // slot e = dispatched_order[i]). After the loop, a final accumulation
    // pass sums these in CANONICAL expert-index order 0..n_expert_used-1 —
    // bit-exact to --uring-projection-overlap's in-loop accumulation order
    // despite the dynamic dispatch.
    //
    // Size = n_expert_used * n_embd floats. Allocated in shared_alloc.
    float * expert_weighted_out;

    // async-experts per-expert intermediate buffers. async-projection-overlap
    // reuses the single-expert up_out / gate_out / act buffers above because
    // it always completes one expert end-to-end before starting the next.
    // async-experts may dispatch matmuls out of expert order (e.g. UP_MATMUL[0]
    // then UP_MATMUL[1] before GATE_MATMUL[0] is loaded), so each expert
    // needs its own slot. Same row-stride convention as the single buffers:
    // index k goes 0..ffn_dim-1, expert E starts at offset E * ffn_dim. Sized
    // n_expert_used * ffn_dim floats each.
    float * ae_up_out_per_expert;
    float * ae_gate_out_per_expert;
    float * ae_act_per_expert;

    // async-experts dispatcher communication: thread 0 picks the next action
    // and writes these two fields at slot[iter & 1]; all threads read from
    // slot[iter & 1] after the (single) inter-iteration barrier.
    // ae_action_type uses the async_experts_action_t enum; sentinel value -1
    // means "all experts done, exit dispatch loop".
    //
    // Why std::atomic<int> with seq_cst:
    //   gdb evidence (Apr 30 2026) of a deadlock from a torn / stale read of
    //   a plain int field. seq_cst on both ends pins ordering to the atomic
    //   itself rather than relying on the surrounding barrier.
    //
    // Why a 2-deep ring indexed by (iter & 1) (May 1 2026):
    //   Originally a single field guarded by a two-barrier handshake
    //   (publish-visible + read-complete) to defeat the publish-overwrite
    //   race: thread 0 racing ahead to iter K+2 and overwriting the field
    //   before a slow worker had loaded iter K+1's value. With one barrier
    //   per iteration the spread between any two threads' iteration counts
    //   is bounded by 1, so two slots — slot[K & 1] for iter K, slot[K+1 & 1]
    //   for iter K+1 — are sufficient. Iter K+2 reuses slot[K & 1], but by
    //   the time thread 0 reaches that publish, the slow worker has already
    //   passed barrier_{K+2}, which means it has already loaded its iter K+1
    //   value from slot[(K+1) & 1] (the OTHER slot). Race-free, and saves
    //   ~13 barriers per layer-call (measured: ~62K barriers per 200-token run).
    std::atomic<int> ae_action_type[2];
    std::atomic<int> ae_action_expert[2];

    // async-experts deadlock diagnostic: per-thread "last point reached"
    // markers, written by each worker thread as it makes progress through
    // compute_async_experts. The moe_barrier watchdog dumps all entries on
    // timeout — the thread whose marker is *behind* the others identifies
    // who got stuck and exactly where. 16 slots to cover up to 16-thread
    // teams (we use 8). volatile + per-thread slot writes means no atomicity
    // needed (each thread writes only its own slot, watchdog read race is
    // acceptable for diagnostic).
    volatile int ae_thread_mark[16];
    volatile int ae_thread_iter[16];

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
    s->expert_weighted_out = (float *) malloc((size_t) n_expert_used * (size_t) n_embd * sizeof(float));
    s->ae_up_out_per_expert   = (float *) malloc((size_t) n_expert_used * (size_t) ffn_dim * sizeof(float));
    s->ae_gate_out_per_expert = (float *) malloc((size_t) n_expert_used * (size_t) ffn_dim * sizeof(float));
    s->ae_act_per_expert      = (float *) malloc((size_t) n_expert_used * (size_t) ffn_dim * sizeof(float));
    s->n_expert_used = n_expert_used;
    s->barrier_count.store(0, std::memory_order_relaxed);
    s->barrier_gen.store(0, std::memory_order_relaxed);
    // Initialize action fields to a sentinel that's not AE_DONE (-1) and not
    // a valid action (0..4), so a stale read from a fresh shared (i.e. before
    // thread 0's first publication) is detectable rather than silently treated
    // as "all done" or as a real action.
    s->ae_action_type  [0].store(-99, std::memory_order_relaxed);
    s->ae_action_type  [1].store(-99, std::memory_order_relaxed);
    s->ae_action_expert[0].store(-99, std::memory_order_relaxed);
    s->ae_action_expert[1].store(-99, std::memory_order_relaxed);
    for (int t = 0; t < 16; t++) {
        s->ae_thread_mark[t] = 0;
        s->ae_thread_iter[t] = 0;
    }
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
    free(s->expert_weighted_out);
    free(s->ae_up_out_per_expert);
    free(s->ae_gate_out_per_expert);
    free(s->ae_act_per_expert);
    delete s;
}

// Barrier across the ggml worker team for this op invocation.
//
// Implementation history & rationale (Apr 30 2026):
//   The first iteration used `#pragma omp barrier` to share libgomp's team
//   barrier with ggml's inter-op barrier. Intermittent deadlocks were observed
//   where 7/8 threads were stuck at this barrier and the 8th had already
//   exited compute_async_experts (gdb-confirmed). Switching to a custom OMP
//   barrier did NOT fix it (same gdb signature). Root cause traced to a
//   non-atomic `sh->ae_action_type` int field that could be torn/stale-read
//   across the barrier — fixed by making it std::atomic<int> with seq_cst
//   (above). With that fix, either barrier implementation is correct.
//
// We use the atomic sense-reversing barrier (no OMP) so that any future
// regression deadlocks visibly inside our own code rather than corrupting
// libgomp's team state and manifesting as a hang in the next op's barrier.
// The watchdog below fires after 30s of no progress so we get a diagnostic
// abort instead of an indefinite hang.
//
// All `nth` threads must reach this call; control-flow divergence across
// threads is undefined behaviour.
//
// Profiling counters (BSC May 1 2026, async-experts vs async-projection-overlap
// gap attribution): g_moe_barrier_count is incremented once per crossing by
// the unique last-arriver thread (no contention beyond the atomic itself).
// g_moe_barrier_wait_ns sums across all waiting threads — each waiter adds
// its own elapsed wait time on exit. Both are relaxed atomics; ordering
// doesn't matter for run-end aggregation.
static std::atomic<uint64_t> g_moe_barrier_count{0};
static std::atomic<uint64_t> g_moe_barrier_wait_ns{0};

extern "C" {
uint64_t llama_moe_pipeline_get_barrier_count(void) {
    return g_moe_barrier_count.load(std::memory_order_relaxed);
}
uint64_t llama_moe_pipeline_get_barrier_wait_ns(void) {
    return g_moe_barrier_wait_ns.load(std::memory_order_relaxed);
}
void llama_moe_pipeline_reset_barrier_counters(void) {
    g_moe_barrier_count.store(0, std::memory_order_relaxed);
    g_moe_barrier_wait_ns.store(0, std::memory_order_relaxed);
}
}

__attribute__((noinline))
static void moe_barrier(moe_pipeline_shared * s, int nth) {
    if (nth <= 1) return;

    const int gen = s->barrier_gen.load(std::memory_order_relaxed);
    if (s->barrier_count.fetch_add(1, std::memory_order_seq_cst) == nth - 1) {
        s->barrier_count.store(0, std::memory_order_relaxed);
        s->barrier_gen.fetch_add(1, std::memory_order_seq_cst);
        g_moe_barrier_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Watchdog: typical barrier wait is microseconds. A wait longer than
    // 30 seconds indicates a real deadlock — abort with diagnostic so we
    // get a usable stack trace and identify which thread is missing.
    const int64_t watchdog_ns = 30LL * 1000LL * 1000LL * 1000LL;
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Watchdog throttle counter: stack-local (NOT thread_local). Reset to 0 on
    // each barrier entry. Was previously `static thread_local`, which forced
    // GCC to emit a __tls_get_addr@plt call on every spin iteration in
    // -fPIC libllama.so — measured at ~26% of total CPU cycles via perf
    // (May 1 2026). Stack-local makes the increment a pure register op.
    uint32_t spin_check = 0;
    while (s->barrier_gen.load(std::memory_order_acquire) == gen) {
        MOE_CPU_RELAX();
        // Cheap timeout check — only do clock_gettime every ~64K spin iters
        // so the spin loop stays hot in the common case.
        if (((++spin_check) & 0xFFFF) == 0) {
            struct timespec t_now;
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            const int64_t elapsed_ns =
                (int64_t)(t_now.tv_sec  - t_start.tv_sec)  * 1000000000LL +
                (int64_t)(t_now.tv_nsec - t_start.tv_nsec);
            if (elapsed_ns > watchdog_ns) {
                fprintf(stderr,
                    "moe_barrier: WATCHDOG fired after %.1fs — deadlock detected.\n"
                    "  gen-at-entry=%d, current_gen=%d, barrier_count=%d, nth=%d\n"
                    "  Per-thread last-progress markers (line # in llama-moe-pipeline.cpp):\n",
                    elapsed_ns / 1e9,
                    gen,
                    s->barrier_gen.load(std::memory_order_relaxed),
                    s->barrier_count.load(std::memory_order_relaxed),
                    nth);
                int min_iter = INT32_MAX, max_iter = -1;
                int stuck_thread = -1, stuck_line = 0, stuck_iter = 0;
                for (int t = 0; t < nth && t < 16; t++) {
                    int tl = s->ae_thread_mark[t];
                    int ti = s->ae_thread_iter[t];
                    fprintf(stderr, "    thread %d: line=%d iter=%d\n", t, tl, ti);
                    if (ti < min_iter) { min_iter = ti; stuck_thread = t; stuck_line = tl; stuck_iter = ti; }
                    if (ti > max_iter) max_iter = ti;
                }
                if (max_iter > min_iter) {
                    fprintf(stderr,
                        "  → STUCK THREAD: ith=%d, last reached line %d in iter %d "
                        "(others at iter %d). The thread is somewhere AFTER line %d but\n"
                        "    BEFORE the next AE_MARK in the source — likely inside a matmul,\n"
                        "    bias-add loop, swiglu, or accumulate loop.\n",
                        stuck_thread, stuck_line, stuck_iter, max_iter, stuck_line);
                } else {
                    fprintf(stderr,
                        "  → All threads at same iter (%d). Race might be in inter-iter\n"
                        "    progression or in moe_barrier itself.\n", min_iter);
                }
                GGML_ABORT("moe_barrier watchdog: 30s deadlock");
            }
        }
    }

    // Spin loop exited normally — accumulate this waiter's elapsed time.
    // Each waiting thread (every thread except the last arriver, which already
    // returned above) adds its own wait, so the global counter is the SUM of
    // wait times across all waiters. Cost: one clock_gettime per wait + one
    // relaxed atomic add. Watchdog t_start is reused.
    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    const int64_t wait_ns =
        (int64_t)(t_end.tv_sec  - t_start.tv_sec)  * 1000000000LL +
        (int64_t)(t_end.tv_nsec - t_start.tv_nsec);
    g_moe_barrier_wait_ns.fetch_add((uint64_t) wait_ns, std::memory_order_relaxed);
}

// --- Shared helpers (used by both compute_async_projection_overlap and compute_async_experts) ---

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

// Row range when thread 0 is excluded from compute (doing I/O wait instead).
// Partitions N over (nth-1) threads indexed by ith=1..nth-1. Thread 0 gets [0,0).
static inline void moe_row_range_skip0(int N, int ith, int nth, int * start, int * end) {
    if (ith == 0 || nth <= 1) {
        *start = *end = 0;
        return;
    }
    const int logical_ith = ith - 1;
    const int logical_nth = nth - 1;
    const int rows_per_thread = (N + logical_nth - 1) / logical_nth;
    *start = std::min(logical_ith * rows_per_thread, N);
    *end   = std::min(*start + rows_per_thread, N);
}

// =============================================================================
// async-projection-overlap: split-tag (upgate/down) + first-ready dynamic dispatch
//
// Tag scheme (n_expert_used == 4 fits LLAMA_IO_URING_MAX_TAGS == 8 exactly):
//   tag 2e     = upgate pair (gate + up) for expert e, 2 reads
//   tag 2e + 1 = down projection for expert e, 1 read
//
// At layer entry, thread 0 submits all 8 tagged reads upfront, then picks
// whichever expert's upgate pair finishes first as the next to compute.
// Down matmul for expert e runs after swiglu; its I/O (tag=2e+1) is waited
// for just before down compute — usually returns instantly because its reads
// have been in flight during all prior compute phases.
// =============================================================================
int llama_moe_pipeline_compute_async_projection_overlap(const struct llama_moe_fused_args * args) {
    if (args == nullptr || args->ebuf == nullptr || args->shared == nullptr) {
        return -1;
    }
    if (args->n_tokens != 1) {
        if (args->ith == 0) {
            fprintf(stderr, "moe-pipeline-async-projection-overlap: n_tokens=%d, only single-token decode supported\n", args->n_tokens);
        }
        return -1;
    }
    if (!args->has_swiglu_oai) {
        if (args->ith == 0) {
            fprintf(stderr, "moe-pipeline-async-projection-overlap: only SWIGLU_OAI activation supported\n");
        }
        return -1;
    }

    const int n_embd        = args->n_embd;
    const int ffn_dim       = args->ffn_dim;
    const int n_expert_used = args->n_expert_used;
    const int ith           = args->ith;
    const int nth           = args->nth;
    moe_pipeline_shared * sh = args->shared;

    // Sanity: n_expert_used * 2 tags must fit LLAMA_IO_URING_MAX_TAGS.
    if (2 * n_expert_used > LLAMA_IO_URING_MAX_TAGS) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline-async-projection-overlap: n_expert_used=%d requires %d tags, max=%d\n",
                    n_expert_used, 2 * n_expert_used, LLAMA_IO_URING_MAX_TAGS);
        }
        return -1;
    }

    if (sh->n_embd != n_embd || sh->ffn_dim != ffn_dim) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline-async-projection-overlap: shared dims mismatch (alloc %d/%d, op %d/%d)\n",
                    sh->n_embd, sh->ffn_dim, n_embd, ffn_dim);
        }
        return -1;
    }

    const ggml_type_traits_cpu * up_traits   = ggml_get_type_traits_cpu((ggml_type) args->W_up_type);
    const ggml_type_traits_cpu * gate_traits = ggml_get_type_traits_cpu((ggml_type) args->W_gate_type);
    const ggml_type_traits_cpu * down_traits = ggml_get_type_traits_cpu((ggml_type) args->W_down_type);
    const ggml_type_traits_cpu * up_qtraits   = ggml_get_type_traits_cpu(up_traits->vec_dot_type);
    const ggml_type_traits_cpu * down_qtraits = ggml_get_type_traits_cpu(down_traits->vec_dot_type);

    if (up_traits->vec_dot_type != gate_traits->vec_dot_type) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline-async-projection-overlap: up/gate vec_dot_type mismatch not supported\n");
        }
        return -1;
    }

    const size_t up_K_bytes   = ggml_row_size((ggml_type) args->W_up_type, n_embd);
    const size_t gate_K_bytes = ggml_row_size((ggml_type) args->W_gate_type, n_embd);
    const size_t down_K_bytes = ggml_row_size((ggml_type) args->W_down_type, ffn_dim);

    // ---- INIT: submit all 8 tagged reads, dispatch first expert ----
    if (ith == 0) {
        for (int e = 0; e < n_expert_used; e++) {
            const int32_t eid = args->selected_experts[e];
            int r;

            if (e == 0) {
                r = llama_uring_expert_buf_load_upgate_async_tagged_bump(
                        args->ebuf, args->layer, eid, /*tag=*/2 * e);
            } else {
                r = llama_uring_expert_buf_load_upgate_async_tagged(
                        args->ebuf, args->layer, eid, /*tag=*/2 * e);
            }
            if (r < 0) {
                fprintf(stderr, "moe-pipeline-async-projection-overlap: upgate submit failed e=%d layer=%d\n",
                        e, args->layer);
            }
            r = llama_uring_expert_buf_load_down_async_tagged(
                    args->ebuf, args->layer, eid, /*tag=*/2 * e + 1);
            if (r < 0) {
                fprintf(stderr, "moe-pipeline-async-projection-overlap: down submit failed e=%d layer=%d\n",
                        e, args->layer);
            }

            // Capture slot pointers (pack layout: proj 0=down, 1=gate, 2=up).
            sh->e_down[e] = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 0)[0];
            sh->e_gate[e] = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 1)[0];
            sh->e_up  [e] = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 2)[0];
        }

        up_qtraits->from_float(args->x_data, sh->x_q, n_embd);
        memset(args->dst_data, 0, sizeof(float) * n_embd);

        // Dispatch first expert: wait for any upgate-pair to finish.
        int upgate_tags[LLAMA_URING_MAX_EXPERTS];
        for (int e = 0; e < n_expert_used; e++) upgate_tags[e] = 2 * e;
        int winner_idx = llama_uring_expert_buf_wait_any_upgate_ready(
                args->ebuf, upgate_tags, n_expert_used);
        if (winner_idx < 0) {
            fprintf(stderr, "moe-pipeline-async-projection-overlap: first dispatch failed layer=%d\n", args->layer);
            winner_idx = 0;  // best-effort fallback
        }
        // tags[i] == 2 * i, so index i maps directly to expert i.
        sh->dispatched_order[0] = winner_idx;
    }
    moe_barrier(sh, nth);

    // ---- COMPUTE LOOP: dynamic dispatch, one expert per iteration ----
    for (int i = 0; i < n_expert_used; i++) {
        const int e = sh->dispatched_order[i];
        const int32_t eid = args->selected_experts[e];

        int i0_ff, i1_ff;
        moe_row_range(ffn_dim, ith, nth, &i0_ff, &i1_ff);

        moe_matvec_range(sh->up_out, sh->e_up[e], sh->x_q,
                         n_embd, i0_ff, i1_ff, up_K_bytes, up_traits->vec_dot);
        if (args->up_bias_data) {
            const float * up_b = (const float *) args->up_bias_data + (size_t) eid * ffn_dim;
            for (int k = i0_ff; k < i1_ff; k++) sh->up_out[k] += up_b[k];
        }

        moe_matvec_range(sh->gate_out, sh->e_gate[e], sh->x_q,
                         n_embd, i0_ff, i1_ff, gate_K_bytes, gate_traits->vec_dot);
        if (args->gate_bias_data) {
            const float * gate_b = (const float *) args->gate_bias_data + (size_t) eid * ffn_dim;
            for (int k = i0_ff; k < i1_ff; k++) sh->gate_out[k] += gate_b[k];
        }

        moe_swiglu_oai_range(sh->act, sh->gate_out, sh->up_out, i0_ff, i1_ff,
                             args->swiglu_alpha, args->swiglu_limit);

        moe_barrier(sh, nth);

        if (ith == 0) {
            down_qtraits->from_float(sh->act, sh->act_q_per_expert, ffn_dim);
        }
        moe_barrier(sh, nth);

        // Wait for this expert's down before down-matmul. Usually returns
        // immediately: the down read has been in flight since layer entry.
        if (ith == 0) {
            int r = llama_uring_expert_buf_wait_expert_tagged(args->ebuf, /*tag=*/2 * e + 1);
            if (r < 0) {
                fprintf(stderr, "moe-pipeline-async-projection-overlap: down wait failed e=%d layer=%d\n",
                        e, args->layer);
            }
        }
        moe_barrier(sh, nth);

        {
            int i0_e, i1_e;
            moe_row_range(n_embd, ith, nth, &i0_e, &i1_e);

            moe_matvec_range(sh->down_out, sh->e_down[e], sh->act_q_per_expert,
                             ffn_dim, i0_e, i1_e, down_K_bytes, down_traits->vec_dot);
            if (args->down_bias_data) {
                const float * down_b = (const float *) args->down_bias_data + (size_t) eid * n_embd;
                for (int k = i0_e; k < i1_e; k++) sh->down_out[k] += down_b[k];
            }
            // Bit-exactness fix: write per-expert weighted output to a
            // dispatch-order-independent buffer. Canonical-order accumulation
            // happens after the compute loop. This matches the
            // --uring-projection-overlap floating-point summation order
            // (expert 0 + expert 1 + expert 2 + expert 3).
            const float w = args->router_weights[e];
            float * ewo_e = sh->expert_weighted_out + (size_t) e * (size_t) n_embd;
            for (int k = i0_e; k < i1_e; k++) ewo_e[k] = w * sh->down_out[k];
        }

        moe_barrier(sh, nth);

        // Dispatch next expert (if any remain): first-ready among un-dispatched.
        if (ith == 0 && i + 1 < n_expert_used) {
            bool dispatched[LLAMA_URING_MAX_EXPERTS] = {false};
            for (int k = 0; k <= i; k++) dispatched[sh->dispatched_order[k]] = true;

            int remaining_tags[LLAMA_URING_MAX_EXPERTS];
            int remaining_exp [LLAMA_URING_MAX_EXPERTS];
            int n_remaining = 0;
            for (int e2 = 0; e2 < n_expert_used; e2++) {
                if (!dispatched[e2]) {
                    remaining_tags[n_remaining] = 2 * e2;
                    remaining_exp [n_remaining] = e2;
                    n_remaining++;
                }
            }

            int winner = llama_uring_expert_buf_wait_any_upgate_ready(
                    args->ebuf, remaining_tags, n_remaining);
            if (winner < 0) {
                fprintf(stderr, "moe-pipeline-async-projection-overlap: dispatch failed i=%d layer=%d\n",
                        i, args->layer);
                winner = 0;  // fallback to first remaining
            }
            sh->dispatched_order[i + 1] = remaining_exp[winner];
        }
        moe_barrier(sh, nth);
    }

    // ---- FINAL ACCUMULATION: canonical expert-index order for bit-exactness ----
    //
    // All experts have been computed and written to expert_weighted_out[e * n_embd + k]
    // = router_weights[e] * down_e[k], with e = original expert slot (0..n_expert_used-1).
    // Sum them in order 0, 1, 2, 3 into dst. This matches v2's accumulation order
    // and yields bit-exact output even though compute ran in dynamic dispatch order.
    {
        int i0_e, i1_e;
        moe_row_range(n_embd, ith, nth, &i0_e, &i1_e);

        for (int k = i0_e; k < i1_e; k++) {
            float sum = 0.0f;
            for (int e = 0; e < n_expert_used; e++) {
                sum += sh->expert_weighted_out[(size_t) e * (size_t) n_embd + (size_t) k];
            }
            args->dst_data[k] = sum;  // dst was memset to 0 at init; overwrite is fine
        }
    }
    // No final barrier needed: each thread wrote its own row slice of dst;
    // ggml's scheduler will barrier at the node boundary before any downstream op.

    return 0;
}

// =============================================================================
// async-experts: per-(expert, projection) tags + eager dispatch
//
// Tag scheme (n_expert_used == 4 fits LLAMA_IO_URING_MAX_TAGS == 16):
//   tag = expert * 3 + proj   (proj 0=down, 1=gate, 2=up — pack-file order)
// 12 tags total for 4 experts.
//
// State machine per expert:
//   loaded_proj[e][p]   becomes true when CQE for tag(e,p) arrives.
//   computed_up[e]       becomes true when UP_MATMUL[e] finishes.
//   computed_gate[e]     becomes true when GATE_MATMUL[e] finishes.
//   computed_swiglu[e]   becomes true when SWIGLU_QUANT[e] finishes.
//   computed_down[e]     becomes true when DOWN_MATMUL[e] finishes.
//
// Eligibility:
//   UP_MATMUL[e]    iff loaded_proj[e][2] && !computed_up[e]
//   GATE_MATMUL[e]  iff loaded_proj[e][1] && !computed_gate[e]
//   SWIGLU_QUANT[e] iff computed_up[e] && computed_gate[e] && !computed_swiglu[e]
//   DOWN_MATMUL[e]  iff loaded_proj[e][0] && computed_swiglu[e] && !computed_down[e]
//
// Picker (simple, deterministic): iterate experts in order 0..n_expert_used-1,
// for each expert try UP, GATE, DOWN, SWIGLU in that priority. First eligible
// wins. This matches the user-specified Q6 policy (phase priority then
// expert-id) with the practical tweak that DOWN ranks above SWIGLU because
// SWIGLU only becomes eligible after both matmuls so it's the cheapest action
// remaining.
//
// Bit-exactness vs async-projection-overlap: per-expert weighted output goes
// to the same expert_weighted_out[e * n_embd + k] buffer that
// async-projection-overlap uses, and the final reduction sums in canonical
// expert-id order. So as long as each matmul produces the same
// floating-point result regardless of dispatch order (which it does — same
// weights, same input, same kernel), the final dst is bit-exact vs
// async-projection-overlap.
// =============================================================================

enum async_experts_action_t {
    AE_DONE             = -1,
    AE_UP_MATMUL        = 0,  // up matmul only (gate not yet computed)
    AE_UP_AND_FINISH    = 1,  // up matmul + swiglu + quantize (gate already computed)
    AE_GATE_MATMUL      = 2,  // gate matmul only (up not yet computed)
    AE_GATE_AND_FINISH  = 3,  // gate matmul + swiglu + quantize (up already computed)
    AE_DOWN_MATMUL      = 4,
};

int llama_moe_pipeline_compute_async_experts(const struct llama_moe_fused_args * args) {
    if (args == nullptr || args->ebuf == nullptr || args->shared == nullptr) {
        return -1;
    }
    if (args->n_tokens != 1) {
        if (args->ith == 0) {
            fprintf(stderr, "moe-pipeline-async-experts: n_tokens=%d, only single-token decode supported\n", args->n_tokens);
        }
        return -1;
    }
    if (!args->has_swiglu_oai) {
        if (args->ith == 0) {
            fprintf(stderr, "moe-pipeline-async-experts: only SWIGLU_OAI activation supported\n");
        }
        return -1;
    }

    const int n_embd        = args->n_embd;
    const int ffn_dim       = args->ffn_dim;
    const int n_expert_used = args->n_expert_used;
    const int ith           = args->ith;
    const int nth           = args->nth;
    moe_pipeline_shared * sh = args->shared;

    // Sanity: n_expert_used * 3 tags must fit LLAMA_IO_URING_MAX_TAGS.
    constexpr int N_PROJ = 3;
    if (n_expert_used * N_PROJ > LLAMA_IO_URING_MAX_TAGS) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline-async-experts: n_expert_used=%d requires %d tags, max=%d\n",
                    n_expert_used, n_expert_used * N_PROJ, LLAMA_IO_URING_MAX_TAGS);
        }
        return -1;
    }

    if (sh->n_embd != n_embd || sh->ffn_dim != ffn_dim) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline-async-experts: shared dims mismatch (alloc %d/%d, op %d/%d)\n",
                    sh->n_embd, sh->ffn_dim, n_embd, ffn_dim);
        }
        return -1;
    }

    const ggml_type_traits_cpu * up_traits   = ggml_get_type_traits_cpu((ggml_type) args->W_up_type);
    const ggml_type_traits_cpu * gate_traits = ggml_get_type_traits_cpu((ggml_type) args->W_gate_type);
    const ggml_type_traits_cpu * down_traits = ggml_get_type_traits_cpu((ggml_type) args->W_down_type);
    const ggml_type_traits_cpu * up_qtraits   = ggml_get_type_traits_cpu(up_traits->vec_dot_type);
    const ggml_type_traits_cpu * down_qtraits = ggml_get_type_traits_cpu(down_traits->vec_dot_type);

    if (up_traits->vec_dot_type != gate_traits->vec_dot_type) {
        if (ith == 0) {
            fprintf(stderr, "moe-pipeline-async-experts: up/gate vec_dot_type mismatch not supported\n");
        }
        return -1;
    }

    const size_t up_K_bytes   = ggml_row_size((ggml_type) args->W_up_type, n_embd);
    const size_t gate_K_bytes = ggml_row_size((ggml_type) args->W_gate_type, n_embd);
    const size_t down_K_bytes = ggml_row_size((ggml_type) args->W_down_type, ffn_dim);

    // ---- INIT: thread 0 submits all 12 tagged reads upfront ----
    // Local state (thread 0 only — picker reads/writes these).
    bool loaded_proj   [LLAMA_URING_MAX_EXPERTS][N_PROJ] = {{false}};
    bool computed_up   [LLAMA_URING_MAX_EXPERTS] = {false};
    bool computed_gate [LLAMA_URING_MAX_EXPERTS] = {false};
    bool computed_swig [LLAMA_URING_MAX_EXPERTS] = {false};
    bool computed_down [LLAMA_URING_MAX_EXPERTS] = {false};

    auto tag_for = [](int e, int p) { return e * N_PROJ + p; };

    if (ith == 0) {
        // GUARD 1: bounds-check selected_experts[] at layer entry.
        // Catches the case where an upstream op corrupted the topk indices
        // (which would otherwise propagate as out-of-range bias offsets and
        // ultimately as a downstream get_rows assert one or more layers later).
        for (int e = 0; e < n_expert_used; e++) {
            const int32_t eid = args->selected_experts[e];
            if (eid < 0 || eid >= args->n_expert) {
                fprintf(stderr,
                    "moe-pipeline-async-experts GUARD1: layer=%d selected_experts[%d]=%d out of [0,%d)\n",
                    args->layer, e, (int)eid, args->n_expert);
                GGML_ABORT("async-experts guard 1: selected_experts out of range");
            }
        }

        // Submit 12 reads. Bump epoch on the very first one.
        for (int e = 0; e < n_expert_used; e++) {
            const int32_t eid = args->selected_experts[e];
            for (int p = 0; p < N_PROJ; p++) {
                int r;
                if (e == 0 && p == 0) {
                    r = llama_uring_expert_buf_load_proj_async_tagged_bump(
                            args->ebuf, args->layer, eid, p, tag_for(e, p));
                } else {
                    r = llama_uring_expert_buf_load_proj_async_tagged(
                            args->ebuf, args->layer, eid, p, tag_for(e, p));
                }
                if (r < 0) {
                    fprintf(stderr, "moe-pipeline-async-experts: submit failed e=%d p=%d layer=%d\n",
                            e, p, args->layer);
                }
            }
            // Capture per-expert slot pointers.
            sh->e_down[e] = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 0)[0];
            sh->e_gate[e] = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 1)[0];
            sh->e_up  [e] = llama_uring_expert_buf_get_slot_ptrs(args->ebuf, 2)[0];
        }

        up_qtraits->from_float(args->x_data, sh->x_q, n_embd);
        memset(args->dst_data, 0, sizeof(float) * n_embd);
    }
    moe_barrier(sh, nth);

    // ---- DISPATCH LOOP: single barrier per iteration (Path A optimization) ----
    //
    // Optimization 1: collapse the 2-barrier (pre-action + post-action) loop
    // into a single barrier per iteration. Thread 0 updates `computed[]` for
    // the action that just ran AND picks the next action BEFORE the barrier.
    // Threads 1-7 spin at the barrier waiting for thread 0. The barrier's
    // release semantics propagate both (a) the action-write-side from prior
    // iterations and (b) the new v4_action_* publication.
    //
    // Optimization 2: SWIGLU_QUANT is no longer a standalone action. The
    // second-arrival matmul (UP_AND_FINISH or GATE_AND_FINISH) does its own
    // matmul + swiglu + quantize inline (with one internal barrier between
    // swiglu and quantize). Saves ~96 dispatch rounds per token.
    int prev_action_type   = -1;  // sentinel: no previous action on first iter
    int prev_action_expert = -1;
    int dbg_round          = 0;
    // 2-deep ring slot index. With one barrier per iteration the spread
    // between any two threads' iteration counts is bounded by 1, so two slots
    // are enough: thread 0 publishes for iter K to slot[K & 1], threads read
    // slot[K & 1], and the next overwrite (iter K+1) targets slot[(K+1) & 1].
    // Iter K+2 reuses slot[K & 1] but the slow worker has already crossed
    // barrier_{K+2} by then, so its load already happened.
    uint32_t my_iter = 0;
    if (ith == 0) AE_DBG("layer=%d enter, n_expert_used=%d", args->layer, n_expert_used);
    while (true) {
        const uint32_t slot = my_iter & 1u;
        // Thread 0: (a) commit the previous action's state, (b) pick next.
        if (ith == 0) {
            dbg_round++;
            // Profiling: timestamps bracket the entire pre-barrier window so
            // we can attribute thread 0's pick latency between pure compute
            // (state machine) and blocking I/O (wait_any_upgate_ready).
            struct timespec t_pick_start;
            clock_gettime(CLOCK_MONOTONIC, &t_pick_start);
            uint64_t io_wait_ns_this_iter = 0;
            // (a) Apply state effects of whichever action ran in the prior iter.
            // First iteration's prev_action_type is -1, so this is a no-op.
            switch (prev_action_type) {
                case AE_UP_MATMUL:        computed_up  [prev_action_expert] = true; break;
                case AE_UP_AND_FINISH:    computed_up  [prev_action_expert] = true;
                                          computed_swig[prev_action_expert] = true; break;
                case AE_GATE_MATMUL:      computed_gate[prev_action_expert] = true; break;
                case AE_GATE_AND_FINISH:  computed_gate[prev_action_expert] = true;
                                          computed_swig[prev_action_expert] = true; break;
                case AE_DOWN_MATMUL:      computed_down[prev_action_expert] = true; break;
                default: break;
            }

            // (b) Pick the next action. Inner loop drains tag completions
            // (blocking on wait_any_upgate_ready when nothing is eligible).
            int picked_type   = -1;
            int picked_expert = -1;
            bool any_unfinished = false;
            while (picked_type < 0) {
                any_unfinished = false;
                for (int e = 0; e < n_expert_used; e++) {
                    if (computed_down[e]) continue;
                    any_unfinished = true;

                    // Up loaded but not computed → choose AND_FINISH variant
                    // when gate is already done so swiglu+quantize fold in.
                    if (loaded_proj[e][2] && !computed_up[e]) {
                        picked_type   = computed_gate[e] ? AE_UP_AND_FINISH : AE_UP_MATMUL;
                        picked_expert = e;
                        break;
                    }
                    // Gate loaded but not computed → symmetric AND_FINISH fold.
                    if (loaded_proj[e][1] && !computed_gate[e]) {
                        picked_type   = computed_up[e] ? AE_GATE_AND_FINISH : AE_GATE_MATMUL;
                        picked_expert = e;
                        break;
                    }
                    // Down ready and swiglu done → final matmul for this expert.
                    if (loaded_proj[e][0] && computed_swig[e] && !computed_down[e]) {
                        picked_type   = AE_DOWN_MATMUL;
                        picked_expert = e;
                        break;
                    }
                }

                if (!any_unfinished) { picked_type = AE_DONE; break; }
                if (picked_type >= 0) break;

                // Nothing eligible: block on wait_any until a tag completes.
                int watch_tags[LLAMA_IO_URING_MAX_TAGS];
                int n_watch = 0;
                for (int e = 0; e < n_expert_used; e++) {
                    for (int p = 0; p < N_PROJ; p++) {
                        if (!loaded_proj[e][p]) {
                            watch_tags[n_watch++] = tag_for(e, p);
                        }
                    }
                }
                if (n_watch == 0) {
                    fprintf(stderr, "moe-pipeline-async-experts: dispatch stuck layer=%d\n", args->layer);
                    picked_type = AE_DONE;
                    break;
                }
                if (kAsyncExpertsDebug) {
                    char buf[256]; int p = 0;
                    for (int i = 0; i < n_watch && p < 200; i++)
                        p += snprintf(buf+p, sizeof(buf)-p, "%d,", watch_tags[i]);
                    AE_DBG("layer=%d round=%d wait_any n_watch=%d tags=[%s]",
                           args->layer, dbg_round, n_watch, buf);
                }
                int64_t t0 = kAsyncExpertsDebug ? ae_dbg_now_us() : 0;
                struct timespec t_io_a, t_io_b;
                clock_gettime(CLOCK_MONOTONIC, &t_io_a);
                int winner = llama_uring_expert_buf_wait_any_upgate_ready(
                        args->ebuf, watch_tags, n_watch);
                clock_gettime(CLOCK_MONOTONIC, &t_io_b);
                io_wait_ns_this_iter +=
                    (uint64_t)((t_io_b.tv_sec  - t_io_a.tv_sec)  * 1000000000LL +
                               (t_io_b.tv_nsec - t_io_a.tv_nsec));
                if (winner < 0) {
                    fprintf(stderr, "moe-pipeline-async-experts: wait_any failed layer=%d round=%d\n",
                            args->layer, dbg_round);
                    picked_type = AE_DONE;
                    break;
                }
                const int won_tag = watch_tags[winner];
                loaded_proj[won_tag / N_PROJ][won_tag % N_PROJ] = true;
                if (kAsyncExpertsDebug) AE_DBG("layer=%d round=%d wait_any_done winner_idx=%d tag=%d e=%d p=%d elapsed_us=%lld",
                                     args->layer, dbg_round, winner, won_tag,
                                     won_tag/N_PROJ, won_tag%N_PROJ,
                                     (long long)(ae_dbg_now_us()-t0));
            }

            // Publish into this iteration's slot. Threads 1-7 read from the
            // SAME slot index after the barrier. The other slot still holds
            // the previous iteration's value (irrelevant — nobody reads it).
            sh->ae_action_type  [slot].store(picked_type,  std::memory_order_seq_cst);
            sh->ae_action_expert[slot].store(picked_expert, std::memory_order_seq_cst);

            if (kAsyncExpertsDebug && picked_type != AE_DONE) {
                static const char * names[] = {"UP","UP_AND_FINISH","GATE","GATE_AND_FINISH","DOWN"};
                AE_DBG("layer=%d round=%d pick action=%s expert=%d (loaded=%d%d%d/%d%d%d/%d%d%d/%d%d%d computed_up=%d%d%d%d gate=%d%d%d%d swig=%d%d%d%d down=%d%d%d%d)",
                       args->layer, dbg_round,
                       (picked_type>=0 && picked_type<5) ? names[picked_type] : "?",
                       picked_expert,
                       loaded_proj[0][0],loaded_proj[0][1],loaded_proj[0][2],
                       loaded_proj[1][0],loaded_proj[1][1],loaded_proj[1][2],
                       loaded_proj[2][0],loaded_proj[2][1],loaded_proj[2][2],
                       loaded_proj[3][0],loaded_proj[3][1],loaded_proj[3][2],
                       computed_up[0],computed_up[1],computed_up[2],computed_up[3],
                       computed_gate[0],computed_gate[1],computed_gate[2],computed_gate[3],
                       computed_swig[0],computed_swig[1],computed_swig[2],computed_swig[3],
                       computed_down[0],computed_down[1],computed_down[2],computed_down[3]);
            }

            // Carry forward for next iteration's state update.
            prev_action_type   = picked_type;
            prev_action_expert = picked_expert;

            // Profiling: total pre-barrier wall-clock spent by thread 0 on
            // this iter, split into pure compute vs blocking I/O.
            struct timespec t_pick_end;
            clock_gettime(CLOCK_MONOTONIC, &t_pick_end);
            const uint64_t total_ns = (uint64_t)(
                (t_pick_end.tv_sec  - t_pick_start.tv_sec)  * 1000000000LL +
                (t_pick_end.tv_nsec - t_pick_start.tv_nsec));
            const uint64_t compute_ns = total_ns > io_wait_ns_this_iter
                                      ? total_ns - io_wait_ns_this_iter : 0;
            g_ae_t0_pick_compute_ns.fetch_add(compute_ns,        std::memory_order_relaxed);
            g_ae_t0_pick_io_wait_ns.fetch_add(io_wait_ns_this_iter, std::memory_order_relaxed);
        }

        // Single barrier per dispatch iteration. The 2-deep ring on
        // ae_action_type / ae_action_expert closes the publish-overwrite race
        // that previously required a second "read-complete" barrier:
        //   - Worst-case spread between fastest and slowest thread's iter
        //     count is 1 (bounded by the barrier).
        //   - Iter K writes slot K&1; iter K+1 writes slot (K+1)&1 (different).
        //   - Iter K+2 reuses slot K&1, but by then the slowest thread has
        //     already crossed barrier_{K+2}, which forced it to load its
        //     iter K+1 value from slot (K+1)&1 BEFORE thread 0 could publish
        //     into slot K&1 again.
        // Saves one barrier per dispatch iteration — measured ~13 barriers
        // per layer-call (~62K barriers per 200-token run).
        AE_MARK(sh, ith, dbg_round);  // about to enter dispatch barrier
        moe_barrier(sh, nth);
        AE_MARK(sh, ith, dbg_round);  // exited; loading from slot[my_iter & 1]

        const int action_type = sh->ae_action_type  [slot].load(std::memory_order_seq_cst);
        const int E           = sh->ae_action_expert[slot].load(std::memory_order_seq_cst);

        if (action_type == AE_DONE) break;

        // Execute action — all 8 threads cooperate.
        switch (action_type) {
            case AE_UP_MATMUL: {
                AE_MARK(sh, ith, dbg_round);  // entering UP_MATMUL
                int i0_ff, i1_ff;
                moe_row_range(ffn_dim, ith, nth, &i0_ff, &i1_ff);
                float * up_e = sh->ae_up_out_per_expert + (size_t) E * ffn_dim;
                moe_matvec_range(up_e, sh->e_up[E], sh->x_q,
                                 n_embd, i0_ff, i1_ff, up_K_bytes, up_traits->vec_dot);
                AE_MARK(sh, ith, dbg_round);  // matmul done
                if (args->up_bias_data) {
                    const int32_t eid = args->selected_experts[E];
                    const float * up_b = (const float *) args->up_bias_data + (size_t) eid * ffn_dim;
                    for (int k = i0_ff; k < i1_ff; k++) up_e[k] += up_b[k];
                }
                AE_MARK(sh, ith, dbg_round);  // bias done
                break;
            }
            case AE_UP_AND_FINISH: {
                AE_MARK(sh, ith, dbg_round);
                int i0_ff, i1_ff;
                moe_row_range(ffn_dim, ith, nth, &i0_ff, &i1_ff);
                float       * up_e   = sh->ae_up_out_per_expert   + (size_t) E * ffn_dim;
                const float * gate_e = sh->ae_gate_out_per_expert + (size_t) E * ffn_dim;
                float       * act_e  = sh->ae_act_per_expert      + (size_t) E * ffn_dim;

                moe_matvec_range(up_e, sh->e_up[E], sh->x_q,
                                 n_embd, i0_ff, i1_ff, up_K_bytes, up_traits->vec_dot);
                AE_MARK(sh, ith, dbg_round);  // up matmul done
                if (args->up_bias_data) {
                    const int32_t eid = args->selected_experts[E];
                    const float * up_b = (const float *) args->up_bias_data + (size_t) eid * ffn_dim;
                    for (int k = i0_ff; k < i1_ff; k++) up_e[k] += up_b[k];
                }
                AE_MARK(sh, ith, dbg_round);  // bias done
                moe_swiglu_oai_range(act_e, gate_e, up_e, i0_ff, i1_ff,
                                     args->swiglu_alpha, args->swiglu_limit);
                AE_MARK(sh, ith, dbg_round);  // swiglu done
                moe_barrier(sh, nth);
                AE_MARK(sh, ith, dbg_round);  // post-internal-barrier
                // Parallel block-aligned quantize. Each block is independent
                // for Q8_0 (own scale + 32 int8s), so distributing blocks
                // across threads is bit-exact vs the previous thread-0-only
                // path. The block boundaries don't match the matmul/swiglu
                // ffn_dim-balanced boundaries, so the internal barrier above
                // is still needed (thread X reads act_e ranges originally
                // written by other threads).
                //
                // Defensive fallback (May 1 2026): if ffn_dim is not exactly
                // divisible by the quantization block size, the trailing
                // partial block cannot be safely split across threads — fall
                // back to thread-0 serial quantize, which handles partial
                // blocks the same way ggml_row_size already accounts for.
                {
                    const int blk = (int) ggml_blck_size((ggml_type) args->W_down_type);
                    if (ffn_dim % blk != 0) {
                        if (ith == 0) {
                            char * act_q_e = sh->act_q_per_expert + (size_t) E * sh->act_q_row_bytes;
                            down_qtraits->from_float(act_e, act_q_e, ffn_dim);
                        }
                    } else {
                        const int n_blocks = ffn_dim / blk;
                        int b0, b1;
                        moe_row_range(n_blocks, ith, nth, &b0, &b1);
                        if (b1 > b0) {
                            const size_t bytes_per_block = sh->act_q_row_bytes / (size_t) n_blocks;
                            char * act_q_e = sh->act_q_per_expert + (size_t) E * sh->act_q_row_bytes;
                            down_qtraits->from_float(
                                act_e + (size_t) b0 * blk,
                                act_q_e + (size_t) b0 * bytes_per_block,
                                (int64_t)(b1 - b0) * blk);
                        }
                    }
                }
                AE_MARK(sh, ith, dbg_round);  // parallel quantize done
                break;
            }
            case AE_GATE_MATMUL: {
                AE_MARK(sh, ith, dbg_round);
                int i0_ff, i1_ff;
                moe_row_range(ffn_dim, ith, nth, &i0_ff, &i1_ff);
                float * gate_e = sh->ae_gate_out_per_expert + (size_t) E * ffn_dim;
                moe_matvec_range(gate_e, sh->e_gate[E], sh->x_q,
                                 n_embd, i0_ff, i1_ff, gate_K_bytes, gate_traits->vec_dot);
                AE_MARK(sh, ith, dbg_round);
                if (args->gate_bias_data) {
                    const int32_t eid = args->selected_experts[E];
                    const float * gate_b = (const float *) args->gate_bias_data + (size_t) eid * ffn_dim;
                    for (int k = i0_ff; k < i1_ff; k++) gate_e[k] += gate_b[k];
                }
                AE_MARK(sh, ith, dbg_round);
                break;
            }
            case AE_GATE_AND_FINISH: {
                AE_MARK(sh, ith, dbg_round);
                int i0_ff, i1_ff;
                moe_row_range(ffn_dim, ith, nth, &i0_ff, &i1_ff);
                const float * up_e   = sh->ae_up_out_per_expert   + (size_t) E * ffn_dim;
                float       * gate_e = sh->ae_gate_out_per_expert + (size_t) E * ffn_dim;
                float       * act_e  = sh->ae_act_per_expert      + (size_t) E * ffn_dim;

                moe_matvec_range(gate_e, sh->e_gate[E], sh->x_q,
                                 n_embd, i0_ff, i1_ff, gate_K_bytes, gate_traits->vec_dot);
                AE_MARK(sh, ith, dbg_round);  // gate matmul done
                if (args->gate_bias_data) {
                    const int32_t eid = args->selected_experts[E];
                    const float * gate_b = (const float *) args->gate_bias_data + (size_t) eid * ffn_dim;
                    for (int k = i0_ff; k < i1_ff; k++) gate_e[k] += gate_b[k];
                }
                AE_MARK(sh, ith, dbg_round);  // bias done
                moe_swiglu_oai_range(act_e, gate_e, up_e, i0_ff, i1_ff,
                                     args->swiglu_alpha, args->swiglu_limit);
                AE_MARK(sh, ith, dbg_round);  // swiglu done
                moe_barrier(sh, nth);
                AE_MARK(sh, ith, dbg_round);  // post-internal-barrier
                // Parallel block-aligned quantize (see AE_UP_AND_FINISH for
                // rationale and the defensive fallback for non-block-aligned ffn_dim).
                {
                    const int blk = (int) ggml_blck_size((ggml_type) args->W_down_type);
                    if (ffn_dim % blk != 0) {
                        if (ith == 0) {
                            char * act_q_e = sh->act_q_per_expert + (size_t) E * sh->act_q_row_bytes;
                            down_qtraits->from_float(act_e, act_q_e, ffn_dim);
                        }
                    } else {
                        const int n_blocks = ffn_dim / blk;
                        int b0, b1;
                        moe_row_range(n_blocks, ith, nth, &b0, &b1);
                        if (b1 > b0) {
                            const size_t bytes_per_block = sh->act_q_row_bytes / (size_t) n_blocks;
                            char * act_q_e = sh->act_q_per_expert + (size_t) E * sh->act_q_row_bytes;
                            down_qtraits->from_float(
                                act_e + (size_t) b0 * blk,
                                act_q_e + (size_t) b0 * bytes_per_block,
                                (int64_t)(b1 - b0) * blk);
                        }
                    }
                }
                AE_MARK(sh, ith, dbg_round);  // parallel quantize done
                break;
            }
            case AE_DOWN_MATMUL: {
                AE_MARK(sh, ith, dbg_round);
                int i0_e, i1_e;
                moe_row_range(n_embd, ith, nth, &i0_e, &i1_e);
                const char * act_q_e = sh->act_q_per_expert + (size_t) E * sh->act_q_row_bytes;
                moe_matvec_range(sh->down_out, sh->e_down[E], act_q_e,
                                 ffn_dim, i0_e, i1_e, down_K_bytes, down_traits->vec_dot);
                AE_MARK(sh, ith, dbg_round);  // down matmul done
                if (args->down_bias_data) {
                    const int32_t eid = args->selected_experts[E];
                    const float * down_b = (const float *) args->down_bias_data + (size_t) eid * n_embd;
                    for (int k = i0_e; k < i1_e; k++) sh->down_out[k] += down_b[k];
                }
                AE_MARK(sh, ith, dbg_round);  // bias done
                const float w = args->router_weights[E];
                float * ewo_e = sh->expert_weighted_out + (size_t) E * (size_t) n_embd;
                for (int k = i0_e; k < i1_e; k++) ewo_e[k] = w * sh->down_out[k];
                AE_MARK(sh, ith, dbg_round);  // accumulate done
                break;
            }
        }
        // No post-action barrier: state update + new pick happen in thread 0
        // before the next iteration's barrier, which provides the sync.
        my_iter++;
    }

    // GUARD 2: dispatch loop exited — verify all experts ran their DOWN_MATMUL.
    // Picker should set AE_DONE only when all computed_down[e] are true (the
    // any_unfinished path), but the error fallbacks (n_watch==0 or
    // wait_any returning -1) ALSO publish AE_DONE without state being complete.
    // If we exit via error fallback, expert_weighted_out[] for un-finished
    // experts holds stale data from the previous decode step → final accumulation
    // produces a corrupted hidden state → downstream layer's topk produces
    // out-of-range indices. Catch it here at the source.
    if (ith == 0) {
        for (int e = 0; e < n_expert_used; e++) {
            if (!computed_down[e]) {
                fprintf(stderr,
                    "moe-pipeline-async-experts GUARD2: layer=%d expert %d not computed_down "
                    "(picker exited early via error fallback)\n",
                    args->layer, e);
                GGML_ABORT("async-experts guard 2: incomplete dispatch");
            }
        }
    }

    // ---- FINAL ACCUMULATION: canonical expert-index order
    // (bit-exact vs --uring-async-projection-overlap) ----
    {
        int i0_e, i1_e;
        moe_row_range(n_embd, ith, nth, &i0_e, &i1_e);
        for (int k = i0_e; k < i1_e; k++) {
            float sum = 0.0f;
            for (int e = 0; e < n_expert_used; e++) {
                sum += sh->expert_weighted_out[(size_t) e * (size_t) n_embd + (size_t) k];
            }
            args->dst_data[k] = sum;
        }

        // GUARD 3: each thread checks its own slice of dst_data is finite.
        // Catches the case where the dispatcher ran a "complete" sequence but
        // produced NaN/Inf — e.g., a stale gate_e read in AE_UP_AND_FINISH due
        // to a missed barrier, or an action-state-machine race that ran a
        // matmul with the wrong slot pointer.
        for (int k = i0_e; k < i1_e; k++) {
            if (!isfinite(args->dst_data[k])) {
                fprintf(stderr,
                    "moe-pipeline-async-experts GUARD3: layer=%d ith=%d dst_data[%d]=%g not finite\n",
                    args->layer, ith, k, (double) args->dst_data[k]);
                GGML_ABORT("async-experts guard 3: non-finite output");
            }
        }
    }

    return 0;
}

#endif // __linux__
