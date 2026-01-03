#include "tensor_trace.h"
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

int main() {
    printf("=== Tensor Trace Test ===\n\n");

    // Test 1: Initialize
    printf("Test 1: Initializing trace system...\n");
    tensor_trace_init("/tmp/test_tensor_trace.bin", 1024 * 1024);  // 1 MB for test
    printf("✅ Init successful\n\n");

    // Test 2: Log some entries
    printf("Test 2: Logging 10 test entries...\n");
    for (int i = 0; i < 10; i++) {
        struct TensorAccessLog entry = {0};
        entry.timestamp_ns = tensor_trace_get_timestamp_ns();
        entry.token_id = i;
        entry.layer_id = i % 4;
        entry.thread_id = tensor_trace_get_thread_id();
        entry.operation_type = 42;  // Dummy operation
        entry.phase = TRACE_PHASE_GENERATE;
        entry.tensor_idx = i;
        entry.tensor_ptr = (uint64_t)(0x1000 + i * 0x100);
        entry.file_offset = i * 1024;
        entry.size_bytes = 4096;

        tensor_trace_log(&entry);
    }
    printf("✅ Logged 10 entries\n\n");

    // Test 3: Shutdown and verify
    printf("Test 3: Shutting down...\n");
    tensor_trace_shutdown();
    printf("✅ Shutdown successful\n\n");

    // Test 4: Verify file exists and size
    printf("Test 4: Verifying output file...\n");
    FILE* f = fopen("/tmp/test_tensor_trace.bin", "rb");
    if (!f) {
        printf("❌ Failed to open output file\n");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    printf("File size: %ld bytes\n", file_size);
    printf("Expected: %lu bytes (10 entries × 64 bytes)\n", 10 * sizeof(struct TensorAccessLog));

    // Read back first entry
    fseek(f, 0, SEEK_SET);
    struct TensorAccessLog read_entry;
    size_t read_count = fread(&read_entry, sizeof(struct TensorAccessLog), 1, f);
    fclose(f);

    if (read_count != 1) {
        printf("❌ Failed to read entry\n");
        return 1;
    }

    printf("First entry read back:\n");
    printf("  token_id: %u\n", read_entry.token_id);
    printf("  layer_id: %u\n", read_entry.layer_id);
    printf("  thread_id: %u\n", read_entry.thread_id);
    printf("  operation_type: %u\n", read_entry.operation_type);
    printf("  file_offset: %lu\n", read_entry.file_offset);

    assert(read_entry.token_id == 0);
    assert(read_entry.layer_id == 0);
    assert(read_entry.file_offset == 0);

    printf("✅ Data verified correctly\n\n");

    // Clean up
    unlink("/tmp/test_tensor_trace.bin");

    printf("=== All Tests Passed! ===\n");
    return 0;
}
