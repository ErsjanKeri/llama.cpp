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

    // Tagged I/O: completions for non-target tags that arrived early.
    // Size = LLAMA_IO_URING_MAX_TAGS (16). Tags 0-7 used by per-expert
    // (upgate-pair, down) split tags in --uring-async-projection-overlap.
    // Tags 0-11 used by per-(expert, projection) tags in --uring-async-experts.
    int parked[LLAMA_IO_URING_MAX_TAGS];
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

// --- Phased I/O ---

// Phase tag is stored in bits 60-63 of user_data (4 bits, up to 16 tags).
// Expanded from 3 bits to 4 bits so v4's per-(expert, projection) scheme fits.
#define PHASE_TAG_SHIFT 60
#define PHASE_TAG_MASK  0xFu

static inline int cqe_phase(const struct io_uring_cqe * cqe) {
    return (int)((cqe->user_data >> PHASE_TAG_SHIFT) & PHASE_TAG_MASK);
}

int llama_io_uring_submit_phased(
        struct llama_io_uring_context        * ctx,
        const struct llama_io_uring_read_req * reqs,
        int                                    n_reqs,
        int                                    phase_tag) {

    if (!ctx || !reqs || n_reqs <= 0) return -1;
    if (phase_tag < 0 || phase_tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    const uint64_t tag_bits = (uint64_t)(phase_tag & PHASE_TAG_MASK) << PHASE_TAG_SHIFT;

    for (int i = 0; i < n_reqs; i++) {
        struct io_uring_sqe * sqe = io_uring_get_sqe(&ctx->ring);
        if (!sqe) {
            io_uring_submit(&ctx->ring);
            sqe = io_uring_get_sqe(&ctx->ring);
            if (!sqe) {
                fprintf(stderr, "llama_io_uring_submit_phased: SQ full\n");
                return -1;
            }
        }

        uint64_t aligned_offset = align_down(reqs[i].file_offset, DIRECT_IO_ALIGN);
        uint64_t end            = align_up(reqs[i].file_offset + reqs[i].size, DIRECT_IO_ALIGN);
        size_t   aligned_size   = (size_t)(end - aligned_offset);

        io_uring_prep_read(sqe, ctx->fd, reqs[i].dst, aligned_size, aligned_offset);
        sqe->user_data = tag_bits | (uint64_t)i;

        ctx->metrics.total_bytes_submitted += aligned_size;
        ctx->metrics.total_bytes_requested += reqs[i].size;
    }

    int submitted = io_uring_submit(&ctx->ring);
    if (submitted < 0) {
        fprintf(stderr, "llama_io_uring_submit_phased: io_uring_submit failed: %s\n",
                strerror(-submitted));
        return -1;
    }

    ctx->metrics.total_submissions += submitted;
    return submitted;
}

int llama_io_uring_wait_phase(
        struct llama_io_uring_context * ctx,
        int                            phase_tag,
        int                            n_expected) {

    if (!ctx || n_expected <= 0) return -1;
    if (phase_tag < 0 || phase_tag >= LLAMA_IO_URING_MAX_TAGS) return -1;

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    const int target = phase_tag & PHASE_TAG_MASK;
    int completed = 0;
    int errors = 0;

    // Consume any parked completions for our phase first.
    int from_parked = ctx->parked[target] < n_expected ? ctx->parked[target] : n_expected;
    completed += from_parked;
    ctx->parked[target] -= from_parked;

    // Drain CQEs until we have enough matching completions.
    while (completed < n_expected) {
        struct io_uring_cqe * cqe;
        int ret = io_uring_wait_cqe(&ctx->ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "llama_io_uring_wait_phase: io_uring_wait_cqe failed: %s\n",
                    strerror(-ret));
            return -1;
        }

        if (cqe->res < 0) {
            fprintf(stderr, "llama_io_uring_wait_phase: read failed: %s\n",
                    strerror(-(cqe->res)));
            errors++;
            ctx->metrics.total_errors++;
        }

        int cqe_tag = cqe_phase(cqe);
        io_uring_cqe_seen(&ctx->ring, cqe);
        ctx->metrics.total_completions++;

        if (cqe_tag == target) {
            completed++;
        } else {
            ctx->parked[cqe_tag]++;
        }
    }

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    ctx->metrics.total_wait_ns += (timespec_to_ns(&t_end) - timespec_to_ns(&t_start));

    return errors > 0 ? -1 : 0;
}

int llama_io_uring_wait_any_tag_ready(
        struct llama_io_uring_context * ctx,
        const int                     * tags,
        const int                     * n_expected_per_tag,
        int                             n_tags) {

    if (!ctx || !tags || !n_expected_per_tag) return -1;
    if (n_tags <= 0 || n_tags > LLAMA_IO_URING_MAX_TAGS) return -1;

    // Validate each tag and required count.
    for (int i = 0; i < n_tags; i++) {
        if (tags[i] < 0 || tags[i] >= LLAMA_IO_URING_MAX_TAGS) return -1;
        if (n_expected_per_tag[i] <= 0) return -1;
    }

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int consumed[LLAMA_IO_URING_MAX_TAGS] = {0};
    int errors = 0;
    int winner = -1;

    // Phase 1: satisfy from parked counts for each watched tag.
    // If a watched tag is already fully parked, pick the FIRST such index as winner.
    for (int i = 0; i < n_tags; i++) {
        const int tag  = tags[i];
        const int need = n_expected_per_tag[i];
        const int take = ctx->parked[tag] < need ? ctx->parked[tag] : need;
        consumed[i] = take;
        ctx->parked[tag] -= take;
        if (winner < 0 && consumed[i] == need) {
            winner = i;
        }
    }

    // Phase 2: drain CQEs until one watched tag reaches its required count.
    // Each wait_cqe has a 30-second watchdog timeout so a lost CQE manifests
    // as an instant diagnostic abort rather than an indefinite process hang.
    // io_uring_wait_cqe_timeout's struct __kernel_timespec uses tv_sec / tv_nsec.
    while (winner < 0) {
        struct io_uring_cqe * cqe;
        struct __kernel_timespec ts = { .tv_sec = 30, .tv_nsec = 0 };
        int ret = io_uring_wait_cqe_timeout(&ctx->ring, &cqe, &ts);
        if (ret == -ETIME) {
            // Watchdog: a CQE never arrived within 30s. Dump the state we know
            // about and return -1. Caller will propagate the failure.
            fprintf(stderr,
                "[v4-watchdog] io_uring_wait_cqe_timeout fired after 30s — likely deadlock.\n"
                "[v4-watchdog]   watched tags: ");
            for (int i = 0; i < n_tags; i++) {
                fprintf(stderr, "%d(need=%d,got=%d) ",
                        tags[i], n_expected_per_tag[i], consumed[i]);
            }
            fprintf(stderr, "\n[v4-watchdog]   parked[]: ");
            for (int t = 0; t < LLAMA_IO_URING_MAX_TAGS; t++) {
                fprintf(stderr, "[%d]=%d ", t, ctx->parked[t]);
            }
            fprintf(stderr,
                "\n[v4-watchdog]   metrics: submissions=%lld completions=%lld errors=%lld\n",
                (long long)ctx->metrics.total_submissions,
                (long long)ctx->metrics.total_completions,
                (long long)ctx->metrics.total_errors);
            return -1;
        }
        if (ret < 0) {
            fprintf(stderr, "llama_io_uring_wait_any_tag_ready: io_uring_wait_cqe failed: %s\n",
                    strerror(-ret));
            return -1;
        }
        if (cqe->res < 0) {
            fprintf(stderr, "llama_io_uring_wait_any_tag_ready: read failed: %s\n",
                    strerror(-(cqe->res)));
            errors++;
            ctx->metrics.total_errors++;
        }
        int cqe_tag = cqe_phase(cqe);
        io_uring_cqe_seen(&ctx->ring, cqe);
        ctx->metrics.total_completions++;

        // Find a watched index whose tag matches and still needs more.
        int matched = -1;
        for (int i = 0; i < n_tags; i++) {
            if (tags[i] == cqe_tag && consumed[i] < n_expected_per_tag[i]) {
                matched = i;
                break;
            }
        }
        if (matched >= 0) {
            consumed[matched]++;
            if (consumed[matched] == n_expected_per_tag[matched]) {
                winner = matched;
            }
        } else {
            // Not watched, or matched tag was already full — park for later.
            ctx->parked[cqe_tag]++;
        }
    }

    // Phase 3: refund losing tags' partial consumes back into parked[], so
    // subsequent wait_phase / wait_any_tag_ready calls see the correct count.
    for (int i = 0; i < n_tags; i++) {
        if (i == winner) continue;
        ctx->parked[tags[i]] += consumed[i];
    }

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    ctx->metrics.total_wait_ns += (timespec_to_ns(&t_end) - timespec_to_ns(&t_start));

    if (errors > 0) return -1;
    return winner;
}

void llama_io_uring_reset_phases(struct llama_io_uring_context * ctx) {
    if (!ctx) return;
    for (int i = 0; i < LLAMA_IO_URING_MAX_TAGS; i++) {
        ctx->parked[i] = 0;
    }
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
