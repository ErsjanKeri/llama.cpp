// BSC thesis: standalone test for llama-io-uring module
// Reads expert weight slices from the GGUF model file via io_uring + O_DIRECT,
// then verifies the data matches a regular pread() for correctness.
//
// Build:
//   g++ -O2 -o test-io-uring tests/test-io-uring.cpp src/llama-io-uring.cpp \
//       -I src/ -luring -std=c++17
//
// Run:
//   ./test-io-uring /path/to/model.gguf

#ifdef __linux__

#include "llama-io-uring.h"

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

// Expert slice parameters for GPT-OSS-20B (from gguf-dump)
// Each expert slice: 4,406,400 bytes (2880 × 2880 MXFP4)
// Layout in file: blk.0.ffn_down_exps starts at offset 13,008,832
static const size_t   EXPERT_STRIDE   = 4406400;
static const uint64_t BLK0_DOWN_START = 13008832;  // first expert of layer 0 down projection

static int64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

// Read via regular pread() for comparison
static int read_reference(int fd, void * buf, size_t size, uint64_t offset) {
    size_t total = 0;
    while (total < size) {
        ssize_t n = pread(fd, (char *)buf + total, size - total, offset + total);
        if (n <= 0) {
            if (n < 0) perror("pread");
            return -1;
        }
        total += n;
    }
    return 0;
}

static int test_single_read(struct llama_io_uring_context * ctx, int ref_fd) {
    printf("--- Test 1: single expert slice read ---\n");

    // Read expert 5 of layer 0 down projection
    const int expert_id = 5;
    const uint64_t offset = BLK0_DOWN_START + expert_id * EXPERT_STRIDE;

    // For O_DIRECT: we need the ALIGNED size since our buffer will receive aligned data
    const size_t aligned_offset = offset & ~511ULL;
    const size_t aligned_end    = ((offset + EXPERT_STRIDE) + 511) & ~511ULL;
    const size_t aligned_size   = aligned_end - aligned_offset;
    const size_t skip           = offset - aligned_offset;  // bytes to skip at start

    // Allocate aligned buffer for io_uring read
    void * uring_buf = llama_io_uring_alloc_buf(aligned_size);
    if (!uring_buf) {
        fprintf(stderr, "  FAIL: alloc_buf returned NULL\n");
        return -1;
    }

    // Allocate regular buffer for reference read
    void * ref_buf = malloc(EXPERT_STRIDE);
    if (!ref_buf) {
        llama_io_uring_free_buf(uring_buf);
        return -1;
    }

    // io_uring read
    struct llama_io_uring_read_req req = {
        /*.file_offset =*/ offset,
        /*.dst         =*/ uring_buf,
        /*.size        =*/ EXPERT_STRIDE,
    };

    int64_t t0 = get_time_ns();
    int ret = llama_io_uring_read_sync(ctx, &req, 1);
    int64_t t1 = get_time_ns();

    if (ret < 0) {
        fprintf(stderr, "  FAIL: io_uring read_sync returned %d\n", ret);
        free(ref_buf);
        llama_io_uring_free_buf(uring_buf);
        return -1;
    }
    printf("  io_uring read: %.2f ms (%.1f MiB/s)\n",
           (t1 - t0) / 1e6,
           (double)EXPERT_STRIDE / ((t1 - t0) / 1e9) / (1024.0 * 1024.0));

    // Reference read via pread
    int64_t t2 = get_time_ns();
    ret = read_reference(ref_fd, ref_buf, EXPERT_STRIDE, offset);
    int64_t t3 = get_time_ns();

    if (ret < 0) {
        fprintf(stderr, "  FAIL: reference pread failed\n");
        free(ref_buf);
        llama_io_uring_free_buf(uring_buf);
        return -1;
    }
    printf("  pread:         %.2f ms (%.1f MiB/s)\n",
           (t3 - t2) / 1e6,
           (double)EXPERT_STRIDE / ((t3 - t2) / 1e9) / (1024.0 * 1024.0));

    // Compare: the io_uring buffer has alignment padding at the start
    // The actual expert data starts at uring_buf + skip
    if (memcmp((char *)uring_buf + skip, ref_buf, EXPERT_STRIDE) != 0) {
        fprintf(stderr, "  FAIL: data mismatch!\n");

        // Find first differing byte for debugging
        const unsigned char * a = (const unsigned char *)uring_buf + skip;
        const unsigned char * b = (const unsigned char *)ref_buf;
        for (size_t i = 0; i < EXPERT_STRIDE; i++) {
            if (a[i] != b[i]) {
                fprintf(stderr, "  First diff at byte %zu: uring=0x%02x pread=0x%02x\n",
                        i, a[i], b[i]);
                break;
            }
        }
        free(ref_buf);
        llama_io_uring_free_buf(uring_buf);
        return -1;
    }
    printf("  Data verification: PASS (all %zu bytes match)\n", EXPERT_STRIDE);

    free(ref_buf);
    llama_io_uring_free_buf(uring_buf);
    return 0;
}

static int test_batch_read(struct llama_io_uring_context * ctx, int ref_fd) {
    printf("--- Test 2: batch read (4 experts × 3 projections = 12 reads) ---\n");

    // Simulate one MoE layer: router selects experts 5, 12, 23, 31
    const int expert_ids[] = {5, 12, 23, 31};
    const int n_experts    = 4;
    const int n_proj       = 3;  // down, gate, up
    const int n_reads      = n_experts * n_proj;

    // Projection base offsets for layer 0 (from gguf-dump)
    const uint64_t proj_starts[] = {
        13008832,   // ffn_down_exps start
        154013632,  // ffn_gate_exps start
        295018432,  // ffn_up_exps start
    };

    // Allocate buffers
    struct llama_io_uring_read_req reqs[12];
    void * uring_bufs[12];
    void * ref_bufs[12];

    for (int p = 0; p < n_proj; p++) {
        for (int e = 0; e < n_experts; e++) {
            int idx = p * n_experts + e;
            uint64_t offset = proj_starts[p] + expert_ids[e] * EXPERT_STRIDE;

            // Aligned size for O_DIRECT
            uint64_t aligned_offset = offset & ~511ULL;
            uint64_t aligned_end    = ((offset + EXPERT_STRIDE) + 511) & ~511ULL;
            size_t   aligned_size   = (size_t)(aligned_end - aligned_offset);

            uring_bufs[idx] = llama_io_uring_alloc_buf(aligned_size);
            ref_bufs[idx]   = malloc(EXPERT_STRIDE);

            if (!uring_bufs[idx] || !ref_bufs[idx]) {
                fprintf(stderr, "  FAIL: allocation failed at idx %d\n", idx);
                return -1;
            }

            reqs[idx].file_offset = offset;
            reqs[idx].dst         = uring_bufs[idx];
            reqs[idx].size        = EXPERT_STRIDE;
        }
    }

    // Batch io_uring read
    int64_t t0 = get_time_ns();
    int ret = llama_io_uring_read_sync(ctx, reqs, n_reads);
    int64_t t1 = get_time_ns();

    if (ret < 0) {
        fprintf(stderr, "  FAIL: batch io_uring read_sync returned %d\n", ret);
        goto cleanup;
    }

    size_t total_bytes;
    total_bytes = (size_t)n_reads * EXPERT_STRIDE;
    printf("  io_uring batch (%d reads, %.1f MiB): %.2f ms (%.1f MiB/s)\n",
           n_reads, total_bytes / (1024.0 * 1024.0),
           (t1 - t0) / 1e6,
           (double)total_bytes / ((t1 - t0) / 1e9) / (1024.0 * 1024.0));

    // Sequential pread reference
    {
        int64_t t2 = get_time_ns();
        for (int i = 0; i < n_reads; i++) {
            if (read_reference(ref_fd, ref_bufs[i], EXPERT_STRIDE, reqs[i].file_offset) < 0) {
                fprintf(stderr, "  FAIL: reference pread %d failed\n", i);
                ret = -1;
                goto cleanup;
            }
        }
        int64_t t3 = get_time_ns();
        printf("  pread sequential (%d reads): %.2f ms (%.1f MiB/s)\n",
               n_reads,
               (t3 - t2) / 1e6,
               (double)total_bytes / ((t3 - t2) / 1e9) / (1024.0 * 1024.0));
    }

    // Verify each read
    {
        int mismatches = 0;
        for (int i = 0; i < n_reads; i++) {
            uint64_t skip = reqs[i].file_offset - (reqs[i].file_offset & ~511ULL);
            if (memcmp((char *)uring_bufs[i] + skip, ref_bufs[i], EXPERT_STRIDE) != 0) {
                fprintf(stderr, "  FAIL: data mismatch at read %d\n", i);
                mismatches++;
            }
        }
        if (mismatches == 0) {
            printf("  Data verification: PASS (all %d reads, %zu bytes each)\n",
                   n_reads, EXPERT_STRIDE);
        } else {
            ret = -1;
        }
    }

cleanup:
    for (int i = 0; i < n_reads; i++) {
        llama_io_uring_free_buf(uring_bufs[i]);
        free(ref_bufs[i]);
    }

    // Print metrics
    struct llama_io_uring_metrics m = llama_io_uring_get_metrics(ctx);
    printf("  Metrics: %lld submissions, %lld completions, %lld bytes submitted, "
           "%.2f ms total wait, %lld errors\n",
           (long long)m.total_submissions, (long long)m.total_completions,
           (long long)m.total_bytes_submitted,
           m.total_wait_ns / 1e6, (long long)m.total_errors);

    return ret;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    printf("Model: %s\n\n", model_path);

    // Initialize io_uring context
    struct llama_io_uring_context * ctx = llama_io_uring_init(model_path, 64);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize io_uring context\n");
        return 1;
    }
    printf("io_uring initialized (ring depth = 64)\n\n");

    // Open regular fd for reference reads
    int ref_fd = open(model_path, O_RDONLY);
    if (ref_fd < 0) {
        fprintf(stderr, "Failed to open reference fd: %s\n", strerror(errno));
        llama_io_uring_cleanup(ctx);
        return 1;
    }

    int failures = 0;

    // Test 1: single expert read
    if (test_single_read(ctx, ref_fd) < 0) failures++;
    printf("\n");

    // Reset metrics between tests
    llama_io_uring_reset_metrics(ctx);

    // Test 2: batch read simulating one MoE layer
    if (test_batch_read(ctx, ref_fd) < 0) failures++;
    printf("\n");

    close(ref_fd);
    llama_io_uring_cleanup(ctx);

    if (failures > 0) {
        printf("FAILED: %d test(s) failed\n", failures);
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}

#else // !__linux__

#include <stdio.h>
int main(void) {
    fprintf(stderr, "io_uring tests are Linux-only\n");
    return 1;
}

#endif
