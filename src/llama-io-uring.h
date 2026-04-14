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
