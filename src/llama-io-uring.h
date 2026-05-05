#pragma once

// BSC thesis: io_uring + O_DIRECT buffer manager for SSD-backed MoE inference
// Provides user-space I/O management for expert and non-expert weight loading,
// bypassing kernel page cache via O_DIRECT and batching reads via io_uring.
//
// Usage:
//   1. llama_io_uring_init()       — open file with O_DIRECT, set up io_uring ring
//   2. llama_io_uring_alloc_buf()  — allocate 512-byte aligned buffer for reads
//   3. llama_io_uring_submit()     — submit one or more reads
//   4. llama_io_uring_wait()       — wait for all submitted reads to complete
//   5. llama_io_uring_free_buf()   — free aligned buffer
//   6. llama_io_uring_cleanup()    — tear down ring and close fd

#ifdef __linux__

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context — holds io_uring ring, file descriptor, and metrics
struct llama_io_uring_context;

// --- Lifecycle ---

// Initialize io_uring context: opens file with O_DIRECT, sets up ring.
// Returns NULL on failure. ring_depth = number of SQ entries (e.g., 64).
struct llama_io_uring_context * llama_io_uring_init(
        const char * filepath,
        int          ring_depth);

// Clean up: drain pending I/O, close ring, close fd, free context.
void llama_io_uring_cleanup(struct llama_io_uring_context * ctx);

// --- Buffer management ---

// Allocate a buffer aligned to 512 bytes (required by O_DIRECT).
// Returns NULL on failure. Caller must free with llama_io_uring_free_buf().
void * llama_io_uring_alloc_buf(size_t size);

// Free a buffer allocated by llama_io_uring_alloc_buf().
void llama_io_uring_free_buf(void * buf);

// --- I/O operations ---

// A single read request descriptor.
struct llama_io_uring_read_req {
    uint64_t file_offset;   // byte offset in GGUF file (will be aligned down to 512)
    void   * dst;           // destination buffer (must be 512-aligned)
    size_t   size;          // number of bytes to read (will be aligned up to 512)
};

// Submit one or more read requests to the io_uring ring.
// Returns the number of SQEs successfully submitted, or -1 on error.
// The reads are NOT yet complete after this call — call llama_io_uring_wait().
int llama_io_uring_submit(
        struct llama_io_uring_context    * ctx,
        const struct llama_io_uring_read_req * reqs,
        int                                n_reqs);

// Wait for exactly n_reqs completions from the CQ ring.
// Returns 0 on success (all reads completed), -1 on error.
// On error, partial results may be in the buffers.
int llama_io_uring_wait(
        struct llama_io_uring_context * ctx,
        int                            n_reqs);

// Convenience: submit + wait in one call.
int llama_io_uring_read_sync(
        struct llama_io_uring_context    * ctx,
        const struct llama_io_uring_read_req * reqs,
        int                                n_reqs);

// --- Tagged I/O (for async pipelining: submit at high QD, wait selectively) ---
//
// Tags encode which logical group a read belongs to. submit_phased() stores
// the tag in the high bits of sqe->user_data. wait_phase() drains CQEs but
// only counts completions matching the requested tag; non-matching CQEs are
// "parked" internally and consumed by a subsequent wait_phase() for another tag.
//
// Tag layout depends on the active flag:
//   --uring-async-projection-overlap : 8 tags = 4 experts x {upgate-pair, down}
//   --uring-async-experts            : 12 tags = 4 experts x {up, gate, down}

// Tag range: 0-15 (4 bits stored in user_data bits 60-63).
// 16 tags lets --uring-async-experts address each (expert, projection)
// pair separately (up to 4 experts x 3 projections = 12 distinct tags).
#define LLAMA_IO_URING_MAX_TAGS     16

// Submit reads tagged with a phase. Same as llama_io_uring_submit but encodes
// the phase in user_data for later filtering by wait_phase.
int llama_io_uring_submit_phased(
        struct llama_io_uring_context        * ctx,
        const struct llama_io_uring_read_req * reqs,
        int                                    n_reqs,
        int                                    phase_tag);

// Wait for n_expected completions with the given phase_tag. Completions from
// other phases that arrive are parked and will be consumed by a later
// wait_phase() call with the matching tag. Returns 0 on success, -1 on error.
int llama_io_uring_wait_phase(
        struct llama_io_uring_context * ctx,
        int                            phase_tag,
        int                            n_expected);

// Reset parked-completion counters. Call once per layer before submitting.
void llama_io_uring_reset_phases(struct llama_io_uring_context * ctx);

// Wait until ONE of the watched tags has received its n_expected CQEs.
// tags[]       : array of LLAMA_IO_URING tag values to watch (each 0..7)
// n_expected_per_tag[] : how many CQEs to wait for on each respective tag
// n_tags       : length of the above arrays (must be <= LLAMA_IO_URING_MAX_TAGS)
//
// Semantics:
//   - Consumes parked CQEs for each watched tag first.
//   - If any watched tag is already at its required count, returns its index
//     without draining the ring.
//   - Otherwise drains CQEs from the ring. CQEs for tags NOT in the watched
//     list are parked for later. CQEs for watched tags increment their count.
//   - First watched tag to reach n_expected_per_tag wins and is returned as
//     the index (0..n_tags-1) in the `tags` array.
//   - Partial consumption for losing tags is refunded to parked[], so
//     subsequent wait_phase / wait_any calls see the correct count.
//
// Returns index into tags[] on success, or -1 on error.
int llama_io_uring_wait_any_tag_ready(
        struct llama_io_uring_context * ctx,
        const int                     * tags,
        const int                     * n_expected_per_tag,
        int                             n_tags);

// --- Metrics ---

struct llama_io_uring_metrics {
    int64_t  total_submissions;     // total SQEs submitted
    int64_t  total_completions;     // total CQEs reaped
    int64_t  total_bytes_submitted; // sum of aligned read sizes
    int64_t  total_bytes_requested; // sum of original (unaligned) sizes
    int64_t  total_wait_ns;         // cumulative time spent in wait()
    int64_t  total_errors;          // CQEs with negative result
};

// Get current metrics (cumulative since init). Thread-safe read.
struct llama_io_uring_metrics llama_io_uring_get_metrics(
        const struct llama_io_uring_context * ctx);

// Reset metrics counters to zero.
void llama_io_uring_reset_metrics(struct llama_io_uring_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // __linux__
