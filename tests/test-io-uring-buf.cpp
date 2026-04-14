// BSC thesis: test for expert buffer manager
// Loads expert slices through the buffer manager and verifies against pread.
//
// Build:
//   g++ -O2 -o test-io-uring-buf tests/test-io-uring-buf.cpp \
//       src/llama-io-uring.cpp src/llama-io-uring-buf.cpp \
//       -I src/ -luring -std=c++17
//
// Run:
//   ./test-io-uring-buf /path/to/model.gguf

#ifdef __linux__

#include "llama-io-uring.h"
#include "llama-io-uring-buf.h"

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

// GPT-OSS-20B parameters
static const int N_LAYERS       = 24;
static const int N_EXPERT       = 32;
static const int N_EXPERT_USED  = 4;
static const int N_PROJ         = 3;   // down, gate, up
static const size_t EXPERT_BYTES = 4406400;  // per expert slice

// Layer 0 projection base offsets (expert 0 starts here)
// From gguf-dump, sorted by actual layer number
static const uint64_t PROJ_OFFSETS[24][3] = {
    // [layer] = { down, gate, up }
    {    13008832,   154013632,   295018432 },  // layer 0
    {   436023232,   577028032,   718032832 },  // layer 1
    {  5089181632,  5230186432,  5371191232 },  // layer 2
    {  7204253632,  7345258432,  7486263232 },  // layer 3
    {  7627268032,  7768272832,  7909277632 },  // layer 4
    {  8050282432,  8191287232,  8332292032 },  // layer 5
    {  8473296832,  8614301632,  8755306432 },  // layer 6
    {  8896311232,  9037316032,  9178320832 },  // layer 7
    {  9319325632,  9460330432,  9601335232 },  // layer 8
    {  9742340032,  9883344832, 10024349632 },  // layer 9
    {   859037632,  1000042432,  1141047232 },  // layer 10
    {  1282052032,  1423056832,  1564061632 },  // layer 11
    {  1705066432,  1846071232,  1987076032 },  // layer 12
    {  2128080832,  2269085632,  2410090432 },  // layer 13
    {  2551095232,  2692100032,  2833104832 },  // layer 14
    {  2974109632,  3115114432,  3256119232 },  // layer 15
    {  3397124032,  3538128832,  3679133632 },  // layer 16
    {  3820138432,  3961143232,  4102148032 },  // layer 17
    {  4243152832,  4384157632,  4525162432 },  // layer 18
    {  4666167232,  4807172032,  4948176832 },  // layer 19
    {  5512196032,  5653200832,  5794205632 },  // layer 20
    {  5935210432,  6076215232,  6217220032 },  // layer 21
    {  6358224832,  6499229632,  6640234432 },  // layer 22
    {  6781239232,  6922244032,  7063248832 },  // layer 23
};

static int read_reference(int fd, void * buf, size_t size, uint64_t offset) {
    size_t total = 0;
    while (total < size) {
        ssize_t n = pread(fd, (char *)buf + total, size - total, offset + total);
        if (n <= 0) return -1;
        total += n;
    }
    return 0;
}

static int test_load_one_layer(struct llama_uring_expert_buf * ebuf, int ref_fd,
                               int layer, const int32_t * expert_ids) {
    printf("  Layer %d, experts [%d,%d,%d,%d]: ",
           layer, expert_ids[0], expert_ids[1], expert_ids[2], expert_ids[3]);

    int ret = llama_uring_expert_buf_load(ebuf, layer, expert_ids);
    if (ret < 0) {
        printf("FAIL (load returned %d)\n", ret);
        return -1;
    }

    // Verify each projection × each expert slot
    void * ref = malloc(EXPERT_BYTES);
    if (!ref) return -1;

    int errors = 0;
    for (int p = 0; p < N_PROJ; p++) {
        void * compact_data = llama_uring_expert_buf_get_data(ebuf, p);
        size_t stride = llama_uring_expert_buf_get_stride(ebuf, p);

        for (int s = 0; s < N_EXPERT_USED; s++) {
            uint64_t file_off = PROJ_OFFSETS[layer][p]
                              + (uint64_t)expert_ids[s] * EXPERT_BYTES;

            if (read_reference(ref_fd, ref, EXPERT_BYTES, file_off) < 0) {
                printf("FAIL (pread failed p=%d s=%d)\n", p, s);
                errors++;
                continue;
            }

            void * slot = (char *)compact_data + (size_t)s * stride;
            if (memcmp(slot, ref, EXPERT_BYTES) != 0) {
                printf("FAIL (mismatch p=%d s=%d eid=%d)\n", p, s, expert_ids[s]);

                // Find first diff
                const unsigned char * a = (const unsigned char *)slot;
                const unsigned char * b = (const unsigned char *)ref;
                for (size_t i = 0; i < EXPERT_BYTES; i++) {
                    if (a[i] != b[i]) {
                        printf("    first diff at byte %zu: got 0x%02x expected 0x%02x\n",
                               i, a[i], b[i]);
                        break;
                    }
                }
                errors++;
            }
        }
    }

    free(ref);

    if (errors == 0) {
        printf("PASS (3 proj × 4 experts = 12 reads verified)\n");
    }
    return errors > 0 ? -1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    printf("Model: %s\n\n", model_path);

    // Initialize io_uring
    struct llama_io_uring_context * io_ctx = llama_io_uring_init(model_path, 64);
    if (!io_ctx) { fprintf(stderr, "Failed to init io_uring\n"); return 1; }

    // Build config
    struct llama_uring_expert_buf_config config;
    memset(&config, 0, sizeof(config));
    config.n_layers      = N_LAYERS;
    config.n_expert      = N_EXPERT;
    config.n_expert_used = N_EXPERT_USED;
    config.n_proj        = N_PROJ;

    for (int p = 0; p < N_PROJ; p++) {
        config.projs[p].expert_bytes = EXPERT_BYTES;
        for (int l = 0; l < N_LAYERS; l++) {
            config.projs[p].file_offsets[l] = PROJ_OFFSETS[l][p];
        }
    }

    // Create buffer manager
    struct llama_uring_expert_buf * ebuf = llama_uring_expert_buf_init(io_ctx, &config);
    if (!ebuf) { fprintf(stderr, "Failed to init expert buf\n"); llama_io_uring_cleanup(io_ctx); return 1; }

    // Open reference fd
    int ref_fd = open(model_path, O_RDONLY);
    if (ref_fd < 0) { perror("open ref"); llama_uring_expert_buf_free(ebuf); llama_io_uring_cleanup(io_ctx); return 1; }

    int failures = 0;

    // Test 1: layer 0 with specific experts
    printf("--- Test 1: single layer load + verify ---\n");
    {
        int32_t ids[] = {5, 12, 23, 31};
        if (test_load_one_layer(ebuf, ref_fd, 0, ids) < 0) failures++;
    }

    // Test 2: layer 23 (last) with different experts
    printf("--- Test 2: last layer, different experts ---\n");
    {
        int32_t ids[] = {0, 7, 15, 28};
        if (test_load_one_layer(ebuf, ref_fd, 23, ids) < 0) failures++;
    }

    // Test 3: middle layer (layer 10 — has unusual file offset due to GGUF alphabetical order)
    printf("--- Test 3: layer 10 (non-sequential GGUF offset) ---\n");
    {
        int32_t ids[] = {1, 10, 20, 30};
        if (test_load_one_layer(ebuf, ref_fd, 10, ids) < 0) failures++;
    }

    // Test 4: simulate 5 consecutive layers (like real inference)
    printf("--- Test 4: 5 consecutive layers (layers 0-4) ---\n");
    {
        int32_t test_ids[][4] = {
            {3, 11, 19, 27},
            {5, 12, 23, 31},
            {0,  8, 16, 24},
            {2, 14, 22, 30},
            {6,  9, 17, 25},
        };
        llama_uring_expert_buf_reset_metrics(ebuf);

        int64_t t0 = clock_ns();
        for (int l = 0; l < 5; l++) {
            if (test_load_one_layer(ebuf, ref_fd, l, test_ids[l]) < 0) failures++;
        }
        int64_t t1 = clock_ns();

        struct llama_uring_expert_buf_metrics m = llama_uring_expert_buf_get_metrics(ebuf);
        printf("  5 layers total: %.2f ms (%.1f MiB/s)\n",
               (t1 - t0) / 1e6,
               m.bytes_to_compact / ((t1 - t0) / 1e9) / (1024.0 * 1024.0));
        printf("  Metrics: %lld loads, %lld reads, disk=%.1f MiB, compact=%.1f MiB, "
               "load=%.2f ms, memcpy=%.2f ms\n",
               (long long)m.loads, (long long)m.reads_submitted,
               m.bytes_from_disk / (1024.0 * 1024.0),
               m.bytes_to_compact / (1024.0 * 1024.0),
               m.load_time_ns / 1e6,
               m.memcpy_time_ns / 1e6);
    }

    close(ref_fd);
    llama_uring_expert_buf_free(ebuf);
    llama_io_uring_cleanup(io_ctx);

    printf("\n");
    if (failures > 0) {
        printf("FAILED: %d test(s) failed\n", failures);
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}

#else
#include <stdio.h>
int main(void) { fprintf(stderr, "Linux only\n"); return 1; }
#endif
