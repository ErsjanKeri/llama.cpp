// BSC thesis: io_uring + O_DIRECT buffer manager implementation
// See llama-io-uring.h for API documentation.

#ifdef __linux__

#include "llama-io-uring.h"

#include <liburing.h>

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>

// O_DIRECT requires 512-byte alignment for offset, buffer address, and size
#define DIRECT_IO_ALIGN 512

struct llama_io_uring_context {
    struct io_uring ring;
    int             fd;           // O_DIRECT file descriptor
    int             ring_depth;
    bool            ring_initialized;

    // cumulative metrics
    struct llama_io_uring_metrics metrics;
};

// --- Helpers ---

static inline uint64_t align_down(uint64_t val, uint64_t align) {
    return val & ~(align - 1);
}

static inline uint64_t align_up(uint64_t val, uint64_t align) {
    return (val + align - 1) & ~(align - 1);
}

static int64_t timespec_to_ns(struct timespec * ts) {
    return (int64_t)ts->tv_sec * 1000000000LL + (int64_t)ts->tv_nsec;
}

// --- Lifecycle ---

struct llama_io_uring_context * llama_io_uring_init(
        const char * filepath,
        int          ring_depth) {

    if (!filepath || ring_depth <= 0) {
        fprintf(stderr, "llama_io_uring_init: invalid arguments\n");
        return NULL;
    }

    // Open with O_DIRECT to bypass page cache
    int fd = open(filepath, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fprintf(stderr, "llama_io_uring_init: open(%s, O_DIRECT) failed: %s\n",
                filepath, strerror(errno));
        return NULL;
    }

    struct llama_io_uring_context * ctx =
        (struct llama_io_uring_context *)calloc(1, sizeof(struct llama_io_uring_context));
    if (!ctx) {
        close(fd);
        return NULL;
    }

    ctx->fd         = fd;
    ctx->ring_depth = ring_depth;

    // Initialize io_uring ring
    int ret = io_uring_queue_init(ring_depth, &ctx->ring, 0);
    if (ret < 0) {
        fprintf(stderr, "llama_io_uring_init: io_uring_queue_init failed: %s\n",
                strerror(-ret));
        close(fd);
        free(ctx);
        return NULL;
    }
    ctx->ring_initialized = true;

    // Register the file descriptor for faster submissions
    ret = io_uring_register_files(&ctx->ring, &ctx->fd, 1);
    if (ret < 0) {
        // Non-fatal: direct fd mode still works, just slower
        fprintf(stderr, "llama_io_uring_init: io_uring_register_files failed: %s (continuing without)\n",
                strerror(-ret));
    }

    memset(&ctx->metrics, 0, sizeof(ctx->metrics));

    return ctx;
}

void llama_io_uring_cleanup(struct llama_io_uring_context * ctx) {
    if (!ctx) return;

    if (ctx->ring_initialized) {
        io_uring_queue_exit(&ctx->ring);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    free(ctx);
}

// --- Buffer management ---

void * llama_io_uring_alloc_buf(size_t size) {
    if (size == 0) return NULL;

    // Round up to 512-byte alignment for O_DIRECT
    size_t aligned_size = (size_t)align_up(size, DIRECT_IO_ALIGN);

    void * buf = NULL;
    int ret = posix_memalign(&buf, DIRECT_IO_ALIGN, aligned_size);
    if (ret != 0) {
        fprintf(stderr, "llama_io_uring_alloc_buf: posix_memalign(%zu) failed: %s\n",
                aligned_size, strerror(ret));
        return NULL;
    }
    return buf;
}

void llama_io_uring_free_buf(void * buf) {
    free(buf);  // posix_memalign'd memory is freed with free()
}

// --- I/O operations ---

int llama_io_uring_submit(
        struct llama_io_uring_context    * ctx,
        const struct llama_io_uring_read_req * reqs,
        int                                n_reqs) {

    if (!ctx || !reqs || n_reqs <= 0) return -1;

    for (int i = 0; i < n_reqs; i++) {
        struct io_uring_sqe * sqe = io_uring_get_sqe(&ctx->ring);
        if (!sqe) {
            fprintf(stderr, "llama_io_uring_submit: SQ full at request %d/%d\n", i, n_reqs);
            // Submit what we have so far, then try again
            io_uring_submit(&ctx->ring);
            sqe = io_uring_get_sqe(&ctx->ring);
            if (!sqe) {
                fprintf(stderr, "llama_io_uring_submit: SQ still full after submit\n");
                return -1;
            }
        }

        // Align offset down and size up for O_DIRECT
        uint64_t orig_offset = reqs[i].file_offset;
        size_t   orig_size   = reqs[i].size;

        uint64_t aligned_offset = align_down(orig_offset, DIRECT_IO_ALIGN);
        uint64_t end            = align_up(orig_offset + orig_size, DIRECT_IO_ALIGN);
        size_t   aligned_size   = (size_t)(end - aligned_offset);

        io_uring_prep_read(sqe, ctx->fd, reqs[i].dst, aligned_size, aligned_offset);
        sqe->user_data = (uint64_t)i;  // tag for identifying in CQ

        ctx->metrics.total_bytes_submitted += aligned_size;
        ctx->metrics.total_bytes_requested += orig_size;
    }

    int submitted = io_uring_submit(&ctx->ring);
    if (submitted < 0) {
        fprintf(stderr, "llama_io_uring_submit: io_uring_submit failed: %s\n",
                strerror(-submitted));
        return -1;
    }

    ctx->metrics.total_submissions += submitted;
    return submitted;
}

int llama_io_uring_wait(
        struct llama_io_uring_context * ctx,
        int                            n_reqs) {

    if (!ctx || n_reqs <= 0) return -1;

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int completed = 0;
    int errors = 0;

    while (completed < n_reqs) {
        struct io_uring_cqe * cqe;
        int ret = io_uring_wait_cqe(&ctx->ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "llama_io_uring_wait: io_uring_wait_cqe failed: %s\n",
                    strerror(-ret));
            return -1;
        }

        if (cqe->res < 0) {
            fprintf(stderr, "llama_io_uring_wait: read failed (req %llu): %s\n",
                    (unsigned long long)cqe->user_data, strerror(-cqe->res));
            errors++;
            ctx->metrics.total_errors++;
        }

        io_uring_cqe_seen(&ctx->ring, cqe);
        completed++;
    }

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    ctx->metrics.total_completions += completed;
    ctx->metrics.total_wait_ns += (timespec_to_ns(&t_end) - timespec_to_ns(&t_start));

    return errors > 0 ? -1 : 0;
}

int llama_io_uring_read_sync(
        struct llama_io_uring_context    * ctx,
        const struct llama_io_uring_read_req * reqs,
        int                                n_reqs) {

    int submitted = llama_io_uring_submit(ctx, reqs, n_reqs);
    if (submitted < 0) return -1;

    return llama_io_uring_wait(ctx, submitted);
}

// --- Metrics ---

struct llama_io_uring_metrics llama_io_uring_get_metrics(
        const struct llama_io_uring_context * ctx) {
    if (!ctx) {
        struct llama_io_uring_metrics empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return ctx->metrics;
}

void llama_io_uring_reset_metrics(struct llama_io_uring_context * ctx) {
    if (ctx) {
        memset(&ctx->metrics, 0, sizeof(ctx->metrics));
    }
}

#endif // __linux__
