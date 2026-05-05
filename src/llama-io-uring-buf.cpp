// BSC thesis: Expert buffer manager implementation (cache-aware)
// See llama-io-uring-buf.h for the API contract.
//
// Architecture:
//   * Single 512-aligned cache buffer of N × slot_bytes bytes.
//   * Cache hits return the existing slot's address (no I/O, no copy).
//   * Cache misses pop the LRU tail, queue a single io_uring read into
//     that slot, and return its address.
//   * mul_mat_id is patched to read each expert via tensor->extra:
//     extra is an array of n_expert_used void* pointers, one per active
//     expert. The pointers can be anywhere in the cache buffer; mul_mat_id
//     no longer assumes constant stride between experts.
//
// When cache_slots == 0 the manager falls back to a "scratch pool" of
// n_proj * n_expert_used slots that are reused every load(). Every access
// is then a fresh io_uring read (no caching), but the data path is still
// zero-copy through the indirection.

#ifdef __linux__

#include "llama-io-uring-buf.h"
#include "llama-io-uring.h"
#include "bsc-expert-pack.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

inline int64_t clock_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

// Encode (layer, proj, expert) into a single uint32_t for hash map keys.
// 16 bits for layer, 8 for proj, 8 for expert — plenty of headroom for any
// model we support (max layer 65535, max proj 255, max expert 255).
inline uint32_t encode_key(uint32_t layer, uint32_t proj, uint32_t expert) {
    return (layer << 16) | (proj << 8) | expert;
}

// Compute the linear index into the frequency table.
inline size_t freq_index(int n_proj, int n_expert,
                         int layer, int proj, int expert) {
    return (size_t)layer * (size_t)n_proj * (size_t)n_expert
         + (size_t)proj * (size_t)n_expert
         + (size_t)expert;
}

}  // namespace

struct llama_uring_expert_buf {
    llama_io_uring_context * io_ctx = nullptr;

    // Config (copied)
    int      n_layers      = 0;
    int      n_expert      = 0;
    int      n_expert_used = 0;
    int      n_proj        = 0;
    uint64_t expert_bytes  = 0;
    uint64_t slot_bytes    = 0;

    int      cache_slots     = 0;       // 0 = no caching
    int      cache_policy    = LLAMA_URING_CACHE_LRU;
    bool     caching_enabled = false;
    int      n_total_slots   = 0;       // cache_slots, or scratch (n_proj * n_expert_used)

    // The single cache buffer (also used as the scratch pool when caching disabled).
    void *  cache_buf  = nullptr;
    size_t  cache_size = 0;

    // Cache state (only meaningful when caching_enabled)
    std::unordered_map<uint32_t, int>          map;          // key -> slot index
    std::vector<uint32_t>                      slot_to_key;  // slot -> encoded key
    std::vector<bool>                          slot_valid;   // slot has data

    // LRU-only state
    std::list<int>                             lru;          // MRU at front, LRU at back
    std::vector<std::list<int>::iterator>      slot_to_iter; // slot -> position in lru

    // LFU / LFU-aging shared state
    std::vector<uint32_t>                      slot_freq;    // per-occupancy access count
    int                                        n_filled = 0; // monotonic during initial fill

    // Per-load "epoch" used by LFU/LFU-aging to prevent picking the same slot
    // as victim twice within a single load() call. Without this, two cache
    // misses in the same load() can both resolve to the same slot index
    // (because the freshly-evicted slot has slot_freq=1, which is often still
    // the global minimum), causing the io_uring batch to issue two reads into
    // the same buffer and corrupting one of them.
    std::vector<uint32_t>                      slot_load_epoch;
    uint32_t                                   current_load_epoch = 0;

    // LFU-aging only: every `aging_period` accesses, halve all slot_freq
    // counters. This prevents the "stale popularity" pathology where experts
    // accessed heavily early in the run (e.g., during prompt eval) accumulate
    // permanently high counters and can never be evicted even when no longer
    // needed. The period is set to 10 × cache_slots, meaning ~10 full cache
    // rotations' worth of accesses before any decay.
    uint32_t                                   aging_counter = 0;
    uint32_t                                   aging_period  = 0;  // 0 = aging disabled (pure LFU)

    // Per-call slot pointers, refilled on every load().
    // Indexed [proj][slot_in_layer]. Returned by get_slot_ptrs().
    void * slot_ptrs[LLAMA_URING_MAX_PROJ][LLAMA_URING_MAX_EXPERTS] = {};

    // 2-phase overlap state: number of async io_uring reads submitted by
    // phase2_submit() that haven't been waited on yet. phase2_wait() drains
    // this. Zero when no async reads are in flight.
    int pending_phase2_reads = 0;

    // Per-tag pending counters for the async pipeline paths
    // (--uring-async-projection-overlap, --uring-async-experts).
    // Tag T counts async reads submitted with tag T (via load_expert_async_tagged
    // or load_expert_async_tagged_pair). Drained by wait_expert_tagged(tag).
    int pending_per_tag[LLAMA_IO_URING_MAX_TAGS] = {0};

    // Global frequency table (incremented on every access, hit or miss).
    std::vector<uint32_t> freq_table;

    llama_uring_expert_buf_metrics metrics = {};
};

// =============================================================================
// Lifecycle
// =============================================================================

llama_uring_expert_buf * llama_uring_expert_buf_init(
        llama_io_uring_context              * io_ctx,
        const llama_uring_expert_buf_config * config) {

    if (!io_ctx || !config) return nullptr;

    if (config->n_layers <= 0 || config->n_layers > LLAMA_URING_MAX_LAYERS) {
        fprintf(stderr, "llama_uring_expert_buf_init: n_layers=%d out of range\n",
                config->n_layers);
        return nullptr;
    }
    if (config->n_expert_used <= 0 || config->n_expert_used > LLAMA_URING_MAX_EXPERTS) {
        fprintf(stderr, "llama_uring_expert_buf_init: n_expert_used=%d out of range\n",
                config->n_expert_used);
        return nullptr;
    }
    if (config->n_proj <= 0 || config->n_proj > LLAMA_URING_MAX_PROJ) {
        fprintf(stderr, "llama_uring_expert_buf_init: n_proj=%d out of range\n",
                config->n_proj);
        return nullptr;
    }
    if (config->expert_bytes == 0 || config->slot_bytes == 0 ||
        config->slot_bytes < config->expert_bytes ||
        (config->slot_bytes % BSC_EXPERT_PACK_ALIGN) != 0) {
        fprintf(stderr,
            "llama_uring_expert_buf_init: bad slot_bytes/expert_bytes "
            "(slot=%lu, expert=%lu)\n",
            (unsigned long)config->slot_bytes,
            (unsigned long)config->expert_bytes);
        return nullptr;
    }

    const int min_cache_for_one_layer = config->n_proj * config->n_expert_used;
    if (config->cache_slots != 0 && config->cache_slots < min_cache_for_one_layer) {
        fprintf(stderr,
            "llama_uring_expert_buf_init: cache_slots=%d is smaller than one "
            "layer's working set (%d). Set cache_slots >= %d or 0 to disable.\n",
            config->cache_slots, min_cache_for_one_layer, min_cache_for_one_layer);
        return nullptr;
    }

    if (config->cache_policy != LLAMA_URING_CACHE_LRU &&
        config->cache_policy != LLAMA_URING_CACHE_LFU &&
        config->cache_policy != LLAMA_URING_CACHE_LFU_AGING) {
        fprintf(stderr,
            "llama_uring_expert_buf_init: cache_policy=%d invalid (LRU=0, LFU=1, LFU_AGING=2)\n",
            config->cache_policy);
        return nullptr;
    }

    auto * buf = new (std::nothrow) llama_uring_expert_buf();
    if (!buf) return nullptr;

    buf->io_ctx          = io_ctx;
    buf->n_layers        = config->n_layers;
    buf->n_expert        = config->n_expert;
    buf->n_expert_used   = config->n_expert_used;
    buf->n_proj          = config->n_proj;
    buf->expert_bytes    = config->expert_bytes;
    buf->slot_bytes      = config->slot_bytes;
    buf->cache_slots     = config->cache_slots;
    buf->cache_policy    = config->cache_policy;
    buf->caching_enabled = (config->cache_slots > 0);

    if (buf->caching_enabled) {
        buf->n_total_slots = config->cache_slots;
    } else {
        buf->n_total_slots = config->n_proj * config->n_expert_used;
    }

    buf->cache_size = (size_t)buf->n_total_slots * (size_t)buf->slot_bytes;
    buf->cache_buf  = llama_io_uring_alloc_buf(buf->cache_size);
    if (!buf->cache_buf) {
        fprintf(stderr,
            "llama_uring_expert_buf_init: cache buffer alloc failed (%zu bytes)\n",
            buf->cache_size);
        delete buf;
        return nullptr;
    }

    if (buf->caching_enabled) {
        buf->slot_to_key.assign(buf->n_total_slots, 0);
        buf->slot_valid.assign(buf->n_total_slots, false);
        buf->map.reserve((size_t)buf->n_total_slots * 2);

        if (buf->cache_policy == LLAMA_URING_CACHE_LRU) {
            buf->slot_to_iter.resize(buf->n_total_slots);
            // Initialize LRU with every slot (all empty); evictions of "empty"
            // slots cost nothing (they aren't in the hash map).
            for (int i = 0; i < buf->n_total_slots; i++) {
                buf->lru.push_back(i);
                auto it = buf->lru.end();
                --it;
                buf->slot_to_iter[i] = it;
            }
        } else { // LFU or LFU-aging
            buf->slot_freq.assign(buf->n_total_slots, 0);
            buf->slot_load_epoch.assign(buf->n_total_slots, 0);
            buf->n_filled           = 0;
            buf->current_load_epoch = 0;
            buf->aging_counter      = 0;
            buf->aging_period       = (buf->cache_policy == LLAMA_URING_CACHE_LFU_AGING)
                                        ? (uint32_t)(config->aging_mult * buf->n_total_slots)
                                        : 0;  // 0 = aging disabled for pure LFU
        }
    }

    // Frequency table: n_layers * n_proj * n_expert entries
    buf->freq_table.assign(
        (size_t)buf->n_layers * (size_t)buf->n_proj * (size_t)buf->n_expert, 0);

    const char * policy_name =
        (buf->cache_policy == LLAMA_URING_CACHE_LFU_AGING) ? "LFU-aging" :
        (buf->cache_policy == LLAMA_URING_CACHE_LFU) ? "LFU" : "LRU";

    fprintf(stderr,
        "llama_uring_expert_buf_init: %d proj, %d experts_used, "
        "expert_bytes=%lu, slot_bytes=%lu, cache_slots=%d (%.1f MiB), "
        "policy=%s, freq_table=%zu entries (%zu bytes)\n",
        buf->n_proj, buf->n_expert_used,
        (unsigned long)buf->expert_bytes,
        (unsigned long)buf->slot_bytes,
        buf->cache_slots,
        buf->cache_size / (1024.0 * 1024.0),
        policy_name,
        buf->freq_table.size(),
        buf->freq_table.size() * sizeof(uint32_t));

    return buf;
}

void llama_uring_expert_buf_free(llama_uring_expert_buf * buf) {
    if (!buf) return;
    llama_io_uring_free_buf(buf->cache_buf);
    delete buf;
}

// =============================================================================
// load() — the hot path
// =============================================================================

// Helper: address of slot N in the cache buffer.
static inline void * slot_addr(llama_uring_expert_buf * buf, int slot_idx) {
    return (char *)buf->cache_buf + (size_t)slot_idx * (size_t)buf->slot_bytes;
}

// LFU victim selection: prefer an unused slot during initial fill,
// otherwise scan slot_freq[] for the index with the smallest counter, EXCLUDING
// any slot already touched in the current load() epoch (those are mid-flight
// in the io_uring batch we're about to submit and must not be reused).
// Ties are broken by lowest index (deterministic). O(N) per call; N is small.
//
// Pre-condition: at least one slot has slot_load_epoch != current_load_epoch.
// Caller guarantees this because cache_slots >= n_proj * n_expert_used (the
// maximum number of slots that can be claimed in one load).
static inline int lfu_pick_victim(llama_uring_expert_buf * buf) {
    if (buf->n_filled < buf->n_total_slots) {
        // Initial fill: hand out the next unused slot. (slot_load_epoch is
        // updated by the caller, so we don't need to skip anything here —
        // unused slots have epoch != current by construction.)
        return buf->n_filled++;
    }
    int      victim    = -1;
    uint32_t min_count = 0;
    for (int i = 0; i < buf->n_total_slots; i++) {
        if (buf->slot_load_epoch[i] == buf->current_load_epoch) {
            continue;  // already claimed this load() — must not reuse
        }
        if (victim < 0 || buf->slot_freq[i] < min_count) {
            min_count = buf->slot_freq[i];
            victim    = i;
        }
    }
    // Pre-condition above guarantees victim >= 0.
    return victim;
}

// Internal helper: process projections [p_start, p_end) for the given layer.
// bump_epoch: if true, increment the LFU epoch (should be true for the first
//             phase of a load, false for subsequent phases sharing the same epoch).
// do_wait:    if true, submit+wait for io_uring reads. If false, submit only
//             and store the count in buf->pending_phase2_reads.
// do_warmup_pad: if true, fill extra slot_ptrs for warmup pass (n_ids > n_expert_used).
static int load_projections(
        llama_uring_expert_buf * buf,
        int                      layer,
        const int32_t          * expert_ids,
        int                      n_ids,
        int                      p_start,
        int                      p_end,
        bool                     bump_epoch,
        bool                     do_wait,
        bool                     do_warmup_pad,
        int                      phase_tag = -1) {

    const int64_t t_start = clock_ns();

    if (bump_epoch && buf->caching_enabled &&
        (buf->cache_policy == LLAMA_URING_CACHE_LFU ||
         buf->cache_policy == LLAMA_URING_CACHE_LFU_AGING)) {
        buf->current_load_epoch += 1;
        if (buf->current_load_epoch == 0) {
            std::fill(buf->slot_load_epoch.begin(), buf->slot_load_epoch.end(), 0);
            buf->current_load_epoch = 1;
        }
    }

    const int n_real = (n_ids < buf->n_expert_used) ? n_ids : buf->n_expert_used;

    struct llama_io_uring_read_req reqs[LLAMA_URING_MAX_PROJ * LLAMA_URING_MAX_EXPERTS];
    int n_reqs = 0;

    for (int p = p_start; p < p_end; p++) {
        for (int s = 0; s < n_real; s++) {
            int32_t eid_raw = expert_ids[s];
            int     eid     = (eid_raw >= 0 && eid_raw < buf->n_expert)
                                  ? eid_raw
                                  : 0;

            buf->freq_table[freq_index(buf->n_proj, buf->n_expert, layer, p, eid)] += 1;

            if (buf->aging_period > 0) {
                buf->aging_counter += 1;
                if (buf->aging_counter >= buf->aging_period) {
                    buf->aging_counter = 0;
                    for (int k = 0; k < buf->n_total_slots; k++) {
                        buf->slot_freq[k] >>= 1;
                    }
                }
            }

            const uint32_t key = encode_key((uint32_t)layer, (uint32_t)p, (uint32_t)eid);
            int slot_idx = -1;
            bool need_io = true;

            if (buf->caching_enabled) {
                auto it = buf->map.find(key);
                if (it != buf->map.end()) {
                    // HIT
                    slot_idx = it->second;
                    if (buf->cache_policy == LLAMA_URING_CACHE_LRU) {
                        buf->lru.splice(buf->lru.begin(), buf->lru,
                                        buf->slot_to_iter[slot_idx]);
                        buf->slot_to_iter[slot_idx] = buf->lru.begin();
                    } else {
                        buf->slot_freq[slot_idx] += 1;
                        buf->slot_load_epoch[slot_idx] = buf->current_load_epoch;
                    }
                    buf->metrics.cache_hits += 1;
                    need_io = false;
                } else {
                    // MISS
                    if (buf->cache_policy == LLAMA_URING_CACHE_LRU) {
                        slot_idx = buf->lru.back();
                    } else {
                        slot_idx = lfu_pick_victim(buf);
                    }

                    if (buf->slot_valid[slot_idx]) {
                        buf->map.erase(buf->slot_to_key[slot_idx]);
                        buf->metrics.cache_evictions += 1;
                    }

                    if (buf->cache_policy == LLAMA_URING_CACHE_LRU) {
                        buf->lru.splice(buf->lru.begin(), buf->lru,
                                        buf->slot_to_iter[slot_idx]);
                        buf->slot_to_iter[slot_idx] = buf->lru.begin();
                    } else {
                        buf->slot_freq[slot_idx]       = 1;
                        buf->slot_load_epoch[slot_idx] = buf->current_load_epoch;
                    }

                    buf->slot_to_key[slot_idx] = key;
                    buf->slot_valid[slot_idx]  = true;
                    buf->map[key]              = slot_idx;
                    buf->metrics.cache_misses += 1;
                }
            } else {
                slot_idx = p * buf->n_expert_used + s;
            }

            void * dst = slot_addr(buf, slot_idx);
            buf->slot_ptrs[p][s] = dst;

            if (need_io) {
                const uint64_t slot_in_pack = bsc_expert_pack_slot_index(
                    (uint32_t)layer, (uint32_t)p, (uint32_t)eid,
                    (uint32_t)buf->n_proj, (uint32_t)buf->n_expert);
                const uint64_t pack_off = bsc_expert_pack_slot_offset(
                    slot_in_pack, buf->slot_bytes);

                reqs[n_reqs].file_offset = pack_off;
                reqs[n_reqs].dst         = dst;
                reqs[n_reqs].size        = (size_t)buf->slot_bytes;
                n_reqs++;
            }
        }
    }

    // Warmup padding (only for full loads, not per-phase)
    if (do_warmup_pad && n_ids > n_real) {
        for (int p = p_start; p < p_end; p++) {
            void * fallback = buf->slot_ptrs[p][0];
            for (int s = n_real; s < n_ids; s++) {
                buf->slot_ptrs[p][s] = fallback;
            }
        }
    }

    // Submit io_uring batch
    if (n_reqs > 0) {
        int submitted;
        if (phase_tag >= 0) {
            submitted = llama_io_uring_submit_phased(buf->io_ctx, reqs, n_reqs, phase_tag);
        } else {
            submitted = llama_io_uring_submit(buf->io_ctx, reqs, n_reqs);
        }
        if (submitted < 0) {
            fprintf(stderr, "load_projections: submit failed\n");
            return -1;
        }

        if (do_wait) {
            int ret;
            if (phase_tag >= 0) {
                ret = llama_io_uring_wait_phase(buf->io_ctx, phase_tag, submitted);
            } else {
                ret = llama_io_uring_wait(buf->io_ctx, submitted);
            }
            if (ret < 0) {
                fprintf(stderr, "load_projections: wait failed\n");
                return -1;
            }
        } else {
            // Async: store pending count for later wait.
            // Tagged submits (phase_tag >= 0) update pending_per_tag[tag];
            // untagged submits update pending_phase2_reads for the legacy API.
            if (phase_tag >= 0 && phase_tag < LLAMA_IO_URING_MAX_TAGS) {
                buf->pending_per_tag[phase_tag] += submitted;
            } else {
                buf->pending_phase2_reads += submitted;
            }
        }

        buf->metrics.reads_submitted += n_reqs;
        buf->metrics.bytes_from_disk += (int64_t)n_reqs * (int64_t)buf->slot_bytes;
    }

    buf->metrics.load_time_ns += (clock_ns() - t_start);

    // BSC: dump full access sequence CSV if env var BSC_CACHE_DUMP is set.
    // Format: load_id,layer,p_start,p_end,n_real,expert_ids...,hits,misses,slot_assignments...
    // This is the ground truth for cache policy simulation validation.
    {
        static FILE * dump_file = nullptr;
        static bool dump_checked = false;
        static int dump_load_id = 0;

        if (!dump_checked) {
            dump_checked = true;
            const char * path = getenv("BSC_CACHE_DUMP");
            if (path && path[0]) {
                dump_file = fopen(path, "w");
                if (dump_file) {
                    fprintf(dump_file, "load_id,layer,p_start,p_end,n_real");
                    for (int i = 0; i < 16; i++) fprintf(dump_file, ",eid%d", i);
                    fprintf(dump_file, ",hits,misses\n");
                    fprintf(stderr, "[BSC_CACHE] Dumping access sequence to %s\n", path);
                }
            }
        }

        if (dump_file) {
            // Count hits/misses from metrics delta
            static int64_t prev_hits = 0, prev_misses = 0;
            int batch_hits = (int)(buf->metrics.cache_hits - prev_hits);
            int batch_misses = (int)(buf->metrics.cache_misses - prev_misses);
            prev_hits = buf->metrics.cache_hits;
            prev_misses = buf->metrics.cache_misses;

            fprintf(dump_file, "%d,%d,%d,%d,%d",
                    dump_load_id, layer, p_start, p_end, n_real);
            for (int i = 0; i < 16; i++) {
                if (i < n_real) {
                    int32_t eid_raw = expert_ids[i];
                    int eid = (eid_raw >= 0 && eid_raw < buf->n_expert) ? eid_raw : 0;
                    fprintf(dump_file, ",%d", eid);
                } else {
                    fprintf(dump_file, ",-1");
                }
            }
            fprintf(dump_file, ",%d,%d\n", batch_hits, batch_misses);
            dump_load_id++;

            // Flush periodically
            if (dump_load_id % 240 == 0) fflush(dump_file);
        }
    }

    return 0;
}

// =============================================================================
// Public load API
// =============================================================================

int llama_uring_expert_buf_load(
        llama_uring_expert_buf * buf,
        int                      layer,
        const int32_t          * expert_ids,
        int                      n_ids) {

    if (!buf || !expert_ids) return -1;
    if (layer < 0 || layer >= buf->n_layers) {
        fprintf(stderr,
            "llama_uring_expert_buf_load: layer=%d out of range [0,%d)\n",
            layer, buf->n_layers);
        return -1;
    }
    if (n_ids <= 0 || n_ids > LLAMA_URING_MAX_EXPERTS) {
        fprintf(stderr,
            "llama_uring_expert_buf_load: n_ids=%d out of range [1,%d]\n",
            n_ids, LLAMA_URING_MAX_EXPERTS);
        return -1;
    }

    buf->metrics.loads += 1;
    return load_projections(buf, layer, expert_ids, n_ids,
                            /*p_start=*/0, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/true, /*do_wait=*/true,
                            /*do_warmup_pad=*/true);
}

// --- 2-phase overlap API ---

int llama_uring_expert_buf_load_phase1(
        llama_uring_expert_buf * buf,
        int                      layer,
        const int32_t          * expert_ids,
        int                      n_ids) {

    if (!buf || !expert_ids) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (n_ids <= 0 || n_ids > LLAMA_URING_MAX_EXPERTS) return -1;

    buf->metrics.loads += 1;
    // Load projections [1, n_proj) = gate + up. Bump epoch. Wait.
    // Pack ordering: 0=down, 1=gate, 2=up. We defer down (index 0) to phase2.
    return load_projections(buf, layer, expert_ids, n_ids,
                            /*p_start=*/1, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/true, /*do_wait=*/true,
                            /*do_warmup_pad=*/true);
}

int llama_uring_expert_buf_load_phase2_submit(
        llama_uring_expert_buf * buf,
        int                      layer,
        const int32_t          * expert_ids,
        int                      n_ids) {

    if (!buf || !expert_ids) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (n_ids <= 0 || n_ids > LLAMA_URING_MAX_EXPERTS) return -1;

    // Load projection [0, 1) = down only. Same epoch. Don't wait.
    // Pack ordering: 0=down. This is the projection we overlap with up+gate compute.
    return load_projections(buf, layer, expert_ids, n_ids,
                            /*p_start=*/0, /*p_end=*/1,
                            /*bump_epoch=*/false, /*do_wait=*/false,
                            /*do_warmup_pad=*/true);
}

int llama_uring_expert_buf_load_phase2_wait(
        llama_uring_expert_buf * buf) {

    if (!buf) return -1;
    if (buf->pending_phase2_reads == 0) return 0;  // all hits, nothing to wait

    const int64_t t_start = clock_ns();
    int ret = llama_io_uring_wait(buf->io_ctx, buf->pending_phase2_reads);
    buf->pending_phase2_reads = 0;
    const int64_t elapsed = clock_ns() - t_start;
    buf->metrics.load_time_ns   += elapsed;
    buf->metrics.phase2_wait_ns += elapsed;

    return (ret < 0) ? -1 : 0;
}

// --- Per-expert async pipelining API ---

int llama_uring_expert_buf_load_expert_sync(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        bool                     bump_epoch) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;

    if (bump_epoch) {
        buf->metrics.loads += 1;
    }

    // Load all 3 projections (down=0, gate=1, up=2) for this single expert.
    // n_ids=1, p_start=0, p_end=n_proj, sync wait.
    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/0, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/bump_epoch, /*do_wait=*/true,
                            /*do_warmup_pad=*/false);
}

int llama_uring_expert_buf_load_expert_async(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;

    // Load all 3 projections for this expert, async (no wait).
    // Same epoch as the sync call that preceded it (no bump).
    // Caller must drain with phase2_wait() before accessing data.
    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/0, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/false, /*do_wait=*/false,
                            /*do_warmup_pad=*/false);
}

int llama_uring_expert_buf_load_expert_async_tagged(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    // Submits all 3 projections for this expert async, tagged.
    // pending_per_tag[tag] += n_reads. Use wait_expert_tagged(tag) to drain.
    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/0, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/false, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

int llama_uring_expert_buf_load_expert_async_tagged_bump(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    buf->metrics.loads += 1;  // first load per layer counts as one "load op"

    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/0, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/true, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

int llama_uring_expert_buf_wait_expert_tagged(
        llama_uring_expert_buf * buf,
        int                      tag) {

    if (!buf) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    const int n = buf->pending_per_tag[tag];
    if (n == 0) return 0;  // all hits for this expert, nothing to wait

    const int64_t t_start = clock_ns();
    int ret = llama_io_uring_wait_phase(buf->io_ctx, tag, n);
    buf->pending_per_tag[tag] = 0;
    const int64_t elapsed = clock_ns() - t_start;
    buf->metrics.load_time_ns   += elapsed;
    buf->metrics.phase2_wait_ns += elapsed;

    return (ret < 0) ? -1 : 0;
}

// --- V3 split-tag helpers ---
//
// Pack layout: projection 0 = down, 1 = gate, 2 = up. upgate = [1, n_proj),
// down = [0, 1). Thin wrappers over load_projections() with narrow p_start/p_end
// and async (do_wait=false). bump_epoch only on the first submission per layer.

int llama_uring_expert_buf_load_upgate_async_tagged(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    // gate + up = projections [1, n_proj). Async, same epoch, tagged.
    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/1, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/false, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

int llama_uring_expert_buf_load_upgate_async_tagged_bump(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    buf->metrics.loads += 1;  // first submission per layer counts as one load op

    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/1, /*p_end=*/buf->n_proj,
                            /*bump_epoch=*/true, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

int llama_uring_expert_buf_load_down_async_tagged(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    // down only = projection [0, 1). Async, same epoch, tagged.
    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/0, /*p_end=*/1,
                            /*bump_epoch=*/false, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

// --- V4 per-(expert, projection) split-tag helpers ---

int llama_uring_expert_buf_load_proj_async_tagged(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      proj,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (proj < 0 || proj >= buf->n_proj) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    // Single projection [proj, proj+1). Async, same epoch, tagged.
    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/proj, /*p_end=*/proj + 1,
                            /*bump_epoch=*/false, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

int llama_uring_expert_buf_load_proj_async_tagged_bump(
        llama_uring_expert_buf * buf,
        int                      layer,
        int32_t                  expert_id,
        int                      proj,
        int                      tag) {

    if (!buf) return -1;
    if (layer < 0 || layer >= buf->n_layers) return -1;
    if (proj < 0 || proj >= buf->n_proj) return -1;
    if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    buf->metrics.loads += 1;  // first submission per layer counts as one load op

    return load_projections(buf, layer, &expert_id, /*n_ids=*/1,
                            /*p_start=*/proj, /*p_end=*/proj + 1,
                            /*bump_epoch=*/true, /*do_wait=*/false,
                            /*do_warmup_pad=*/false,
                            /*phase_tag=*/tag);
}

int llama_uring_expert_buf_wait_any_upgate_ready(
        llama_uring_expert_buf * buf,
        const int              * upgate_tags,
        int                      n_tags) {

    if (!buf || !upgate_tags) return -1;
    if (n_tags <= 0 || n_tags > LLAMA_IO_URING_MAX_TAGS) return -1;

    // Validate tags and build n_expected from current pending counts.
    int n_expected[LLAMA_IO_URING_MAX_TAGS];
    for (int i = 0; i < n_tags; i++) {
        const int tag = upgate_tags[i];
        if (tag < 0 || tag >= LLAMA_IO_URING_MAX_TAGS) return -1;
        n_expected[i] = buf->pending_per_tag[tag];
    }

    // Shortcut: if ANY tag has 0 pending (all cache hits for that upgate pair),
    // it's already fully satisfied — no syscall needed, return that index.
    for (int i = 0; i < n_tags; i++) {
        if (n_expected[i] == 0) {
            return i;
        }
    }

    const int64_t t_start = clock_ns();
    int winner = llama_io_uring_wait_any_tag_ready(
        buf->io_ctx, upgate_tags, n_expected, n_tags);
    const int64_t elapsed = clock_ns() - t_start;
    buf->metrics.load_time_ns   += elapsed;
    buf->metrics.phase2_wait_ns += elapsed;

    if (winner < 0) return -1;

    // Only the winning tag's pending is "claimed" by this call.
    // Losing tags keep their pending count (their partial progress
    // was refunded to parked[] inside wait_any_tag_ready, and will be
    // consumed by a future wait_any / wait_phase).
    buf->pending_per_tag[upgate_tags[winner]] = 0;
    return winner;
}

// =============================================================================
// Access
// =============================================================================

const void * const * llama_uring_expert_buf_get_slot_ptrs(
        const llama_uring_expert_buf * buf, int proj_idx) {
    if (!buf || proj_idx < 0 || proj_idx >= buf->n_proj) return nullptr;
    return (const void * const *) buf->slot_ptrs[proj_idx];
}

size_t llama_uring_expert_buf_get_stride(
        const llama_uring_expert_buf * buf, int proj_idx) {
    if (!buf || proj_idx < 0 || proj_idx >= buf->n_proj) return 0;
    return (size_t)buf->slot_bytes;
}

// =============================================================================
// Metrics
// =============================================================================

llama_uring_expert_buf_metrics llama_uring_expert_buf_get_metrics(
        const llama_uring_expert_buf * buf) {
    if (!buf) return {};
    return buf->metrics;
}

void llama_uring_expert_buf_reset_metrics(llama_uring_expert_buf * buf) {
    if (!buf) return;
    buf->metrics = {};
}

// =============================================================================
// Frequency table
// =============================================================================

const uint32_t * llama_uring_expert_buf_get_freq_table(
        const llama_uring_expert_buf * buf) {
    if (!buf) return nullptr;
    return buf->freq_table.data();
}

size_t llama_uring_expert_buf_get_freq_table_size(
        const llama_uring_expert_buf * buf) {
    if (!buf) return 0;
    return buf->freq_table.size();
}

#endif  // __linux__
