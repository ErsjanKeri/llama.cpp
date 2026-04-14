#pragma once

// BSC thesis: Expert buffer manager for io_uring + O_DIRECT (cache-aware mode)
//
// Reads expert weight slices from a .bscexp pack file via io_uring + O_DIRECT
// into a single 512-aligned cache buffer. The cache holds N slots; each slot
// holds exactly one expert slice (slot_bytes from the pack header).
//
// On every call to llama_uring_expert_buf_load(layer, expert_ids):
//   - For each (layer, projection, expert_id) needed:
//       HIT  → record the slot's address in slot_ptrs[proj][slot_in_layer]
//       MISS → evict the LRU slot, queue an io_uring read into it, record
//              the slot's address in slot_ptrs[proj][slot_in_layer]
//   - Submit io_uring batch (only the misses)
//   - Wait for completions
//
// After load() returns, the caller plugs slot_ptrs[proj] (an array of
// n_expert_used void* pointers) into the corresponding compact tensor's
// `extra` field. mul_mat_id (patched) uses tensor->extra to address each
// expert directly:
//
//   src0_cur = (const char *) ((const void * const *) src0->extra)[cur_a];
//
// This eliminates ALL CPU-driven data motion: cache hits stay where they are,
// cache misses go straight from SSD into the cache via io_uring DMA, and
// mul_mat_id reads from each expert's cache slot regardless of layout.
//
// Cache slot count is configurable. cache_slots = 0 keeps the previous
// pack-only zero-copy path: every call allocates fresh slot pointers but
// every access is treated as a miss (no cache, full I/O on every load).
// cache_slots > 0 enables the LRU.

#ifdef __linux__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_io_uring_context;  // forward decl from llama-io-uring.h

// Compile-time bounds.
//
// LLAMA_URING_MAX_EXPERTS must be >= the model's n_expert (NOT n_expert_used)
// because llama.cpp's warmup pass sets n_expert_used := n_expert temporarily,
// causing the compact tensor to be created with ne[2] = n_expert and
// mul_mat_id to iterate cur_a from 0 to n_expert-1. The slot_ptrs array
// must be large enough to satisfy that worst case.
#define LLAMA_URING_MAX_LAYERS   128
#define LLAMA_URING_MAX_PROJ       4
#define LLAMA_URING_MAX_EXPERTS  256  // worst-case n_expert across supported MoE models

// --- Cache eviction policy ---
//
// LRU: classic least-recently-used (doubly-linked list, O(1) per access).
// LFU: least-frequently-used. Each slot has a per-occupancy access counter
//      that is reset to 1 when a slot is reassigned to a new expert. On a
//      miss, an unused slot is preferred; otherwise the slot with the lowest
//      counter is evicted (linear scan, O(N) per eviction; N is small —
//      hundreds of slots — so cost is negligible vs the I/O latency hidden
//      behind every miss).
//
// Both policies use the same hash map for O(1) lookup; only the eviction
// strategy differs.
enum llama_uring_cache_policy {
    LLAMA_URING_CACHE_LRU       = 0,
    LLAMA_URING_CACHE_LFU       = 1,
    LLAMA_URING_CACHE_LFU_AGING = 2,  // LFU with periodic counter decay (halves all slot_freq every aging_period accesses)
};

// --- Configuration ---
//
// All numeric fields come from the .bscexp pack header that the caller
// already read+validated.

struct llama_uring_expert_buf_config {
    int      n_layers;       // matches pack header
    int      n_expert;       // total experts per layer
    int      n_expert_used;  // experts active per token (from model hparams)
    int      n_proj;         // 2 (down,up) or 3 (down,gate,up)

    uint64_t expert_bytes;   // exact bytes per expert slice
    uint64_t slot_bytes;     // 512-aligned slot stride from pack header

    // Cache configuration
    int      cache_slots;    // size of cache; 0 = caching disabled
    int      cache_policy;   // llama_uring_cache_policy enum value
    int      aging_mult;     // LFU-aging period = aging_mult × cache_slots (default: 10)
};

// --- Opaque handle ---

struct llama_uring_expert_buf;

// --- Lifecycle ---

// Create the buffer manager.
//   - If cache_slots == 0: behaves like the previous pack-file zero-copy
//     path. A small "scratch" pool of n_proj * n_expert_used slots is
//     allocated and reused on every load (no caching across loads).
//   - If cache_slots > 0:  allocates cache_slots slots in a single
//     512-aligned region, builds a hash map and an LRU list, and tracks
//     hit/miss/eviction metrics across the run.
// Returns NULL on failure.
struct llama_uring_expert_buf * llama_uring_expert_buf_init(
        struct llama_io_uring_context              * io_ctx,
        const struct llama_uring_expert_buf_config * config);

// Free the cache region and the manager.
void llama_uring_expert_buf_free(struct llama_uring_expert_buf * buf);

// --- Runtime: load experts for one layer ---

// Look up (and load on miss) the slices needed for `layer`.
// expert_ids: array of `n_ids` real expert IDs (e.g. {5,12,23,31}).
// n_ids: usually n_expert_used (4) during normal inference; can be n_expert
//        during the warmup pass where llama.cpp inflates it.
//
// After this call returns 0, llama_uring_expert_buf_get_slot_ptrs(p) returns
// an array of n_ids void* pointers, one per slot, pointing at the cache slot
// that holds that expert's data. The first n_expert_used pointers are real
// loads; any extra entries (during warmup) are filled with valid fallback
// pointers so mul_mat_id can iterate the inflated n_as without segfaulting.
//
// Returns 0 on success, -1 on error.
int llama_uring_expert_buf_load(
        struct llama_uring_expert_buf * buf,
        int                             layer,
        const int32_t                 * expert_ids,
        int                             n_ids);

// --- Runtime: 2-phase load for I/O-compute overlap ---
//
// Splits the per-layer load into two phases so that the down-projection
// reads can overlap with the up+gate compute:
//
//   phase1_load():    load up+gate projections (bumps epoch, submits, waits)
//   phase2_submit():  submit down projection reads (same epoch, async — NO wait)
//   phase2_wait():    wait for the async down reads to complete
//
// Between phase2_submit() and phase2_wait(), the CPU runs up+gate mul_mat_id
// and swiglu while the SSD reads down weights in parallel.
//
// The shared epoch guarantees that phase2's eviction cannot touch phase1's
// in-use slots (LFU: epoch check; LRU: MRU placement geometry).

int llama_uring_expert_buf_load_phase1(
        struct llama_uring_expert_buf * buf,
        int                             layer,
        const int32_t                 * expert_ids,
        int                             n_ids);

int llama_uring_expert_buf_load_phase2_submit(
        struct llama_uring_expert_buf * buf,
        int                             layer,
        const int32_t                 * expert_ids,
        int                             n_ids);

int llama_uring_expert_buf_load_phase2_wait(
        struct llama_uring_expert_buf * buf);

// --- Access ---

// Get the array of n_expert_used void* slot pointers for projection p.
// Caller writes this pointer into the compact tensor's `extra` field.
// Lifetime: valid until the next call to llama_uring_expert_buf_load().
const void * const * llama_uring_expert_buf_get_slot_ptrs(
        const struct llama_uring_expert_buf * buf,
        int                                   proj_idx);

// Stride to use for the compact tensor's nb[2] (== slot_bytes). This is no
// longer read by mul_mat_id (the patched version uses tensor->extra) but
// remains correct for any other code path that inspects nb[2].
size_t llama_uring_expert_buf_get_stride(
        const struct llama_uring_expert_buf * buf,
        int                                   proj_idx);

// --- Metrics ---

struct llama_uring_expert_buf_metrics {
    int64_t loads;             // number of load() calls
    int64_t reads_submitted;   // total io_uring reads (= cache misses + cold scratch reads)
    int64_t bytes_from_disk;   // total bytes read from SSD

    // Cache-specific metrics (zero when cache_slots == 0)
    int64_t cache_hits;
    int64_t cache_misses;
    int64_t cache_evictions;

    int64_t load_time_ns;      // cumulative time in load() (lookup + submit + wait)
};

struct llama_uring_expert_buf_metrics llama_uring_expert_buf_get_metrics(
        const struct llama_uring_expert_buf * buf);

void llama_uring_expert_buf_reset_metrics(
        struct llama_uring_expert_buf * buf);

// --- Frequency table ---
//
// The buffer manager tracks per-(layer, proj, expert) access counts in a
// global table of size n_layers*n_proj*n_expert*sizeof(uint32_t). This is
// independent of the cache eviction policy: counters are incremented on
// every access (hit and miss). The table is read-only from the caller's
// perspective and lives for the lifetime of the buffer manager.

const uint32_t * llama_uring_expert_buf_get_freq_table(
        const struct llama_uring_expert_buf * buf);

// Total entries in the frequency table.
size_t llama_uring_expert_buf_get_freq_table_size(
        const struct llama_uring_expert_buf * buf);

#ifdef __cplusplus
}
#endif

#endif // __linux__
