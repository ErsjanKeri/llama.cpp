#include "tensor_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#ifdef __linux__
#include <sys/syscall.h>
#endif

// === Global State ===

// Memory-mapped log buffer
static void* g_log_buffer = NULL;
static size_t g_log_capacity = 0;      // Total capacity in bytes
static size_t g_log_offset = 0;        // Current write offset (not thread-safe, needs atomics later)
static int g_log_fd = -1;              // File descriptor for log file
static uint64_t g_trace_start_ns = 0;  // Trace start time (for relative timestamps)

// Thread-local buffer for batching writes (avoid contention)
#define THREAD_LOCAL_BUFFER_SIZE 1024  // 1024 entries = 128KB per thread (128 bytes each)
static __thread struct TensorAccessLog g_thread_local_buffer[THREAD_LOCAL_BUFFER_SIZE];
static __thread size_t g_thread_local_offset = 0;

// === Tensor Registration Table (Path B) ===

#define MAX_REGISTERED_TENSORS 1024  // TinyLlama has ~201 tensors, use 1024 for safety

struct TensorRegistryEntry {
    void* data_ptr;          // Memory address (key for lookup)
    char name[64];           // Tensor name (e.g., "blk.5.attn_q.weight")
    uint64_t file_offset;    // Offset in GGUF file
    uint64_t size_bytes;     // Size in bytes
    uint16_t layer_id;       // Extracted layer ID
    uint32_t tensor_idx;     // Index in registry (for trace logs)
};

static struct TensorRegistryEntry g_tensor_registry[MAX_REGISTERED_TENSORS];
static uint32_t g_registry_count = 0;  // Number of registered tensors

// === Helper Functions ===

// Get current timestamp in nanoseconds (relative to trace start)
uint64_t tensor_trace_get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    return now - g_trace_start_ns;  // Return relative time since trace start
}

// Get current thread ID (Linux-specific)
uint16_t tensor_trace_get_thread_id(void) {
#ifdef __linux__
    return (uint16_t)syscall(SYS_gettid);
#else
    // Fallback for non-Linux (not perfect, but works for debugging)
    return (uint16_t)((uintptr_t)pthread_self() & 0xFFFF);
#endif
}

// === Core API Implementation ===

void tensor_trace_init(const char* log_path, size_t capacity_bytes) {
    if (g_log_buffer != NULL) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Already initialized\n");
        return;
    }

    // Record trace start time
    g_trace_start_ns = tensor_trace_get_timestamp_ns();

    // Open/create log file
    g_log_fd = open(log_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (g_log_fd < 0) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to open %s: %s\n",
                log_path, strerror(errno));
        return;
    }

    // Resize file to capacity
    if (ftruncate(g_log_fd, capacity_bytes) != 0) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to resize file: %s\n",
                strerror(errno));
        close(g_log_fd);
        g_log_fd = -1;
        return;
    }

    // Memory-map the file
    g_log_buffer = mmap(NULL, capacity_bytes, PROT_WRITE, MAP_SHARED, g_log_fd, 0);
    if (g_log_buffer == MAP_FAILED) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to mmap: %s\n",
                strerror(errno));
        close(g_log_fd);
        g_log_fd = -1;
        g_log_buffer = NULL;
        return;
    }

    g_log_capacity = capacity_bytes;
    g_log_offset = 0;

    printf("[TENSOR_TRACE] Initialized: %s (%.2f GB capacity)\n",
           log_path, capacity_bytes / (1024.0 * 1024.0 * 1024.0));
}

void tensor_trace_log(const struct TensorAccessLog* entry) {
    if (g_log_buffer == NULL) {
        return;  // Not initialized, silently skip
    }

    // Add to thread-local buffer
    g_thread_local_buffer[g_thread_local_offset++] = *entry;

    // Flush when buffer is full
    if (g_thread_local_offset >= THREAD_LOCAL_BUFFER_SIZE) {
        // TODO: Add atomic operation here for thread safety
        // For MVP, we'll use simple (non-thread-safe) flush

        size_t bytes_to_write = g_thread_local_offset * sizeof(struct TensorAccessLog);

        // Check capacity
        if (g_log_offset + bytes_to_write <= g_log_capacity) {
            // Copy thread-local buffer to global mmap'd buffer
            memcpy((char*)g_log_buffer + g_log_offset,
                   g_thread_local_buffer,
                   bytes_to_write);
            g_log_offset += bytes_to_write;
        } else {
            fprintf(stderr, "[TENSOR_TRACE] Warning: Log buffer full, dropping entries\n");
        }

        // Reset thread-local buffer
        g_thread_local_offset = 0;
    }
}

void tensor_trace_shutdown(void) {
    if (g_log_buffer == NULL) {
        return;  // Not initialized
    }

    // Flush remaining thread-local entries
    if (g_thread_local_offset > 0) {
        size_t bytes_to_write = g_thread_local_offset * sizeof(struct TensorAccessLog);

        if (g_log_offset + bytes_to_write <= g_log_capacity) {
            memcpy((char*)g_log_buffer + g_log_offset,
                   g_thread_local_buffer,
                   bytes_to_write);
            g_log_offset += bytes_to_write;
        }

        g_thread_local_offset = 0;
    }

    // Sync to disk
    msync(g_log_buffer, g_log_capacity, MS_SYNC);

    // Unmap and close
    munmap(g_log_buffer, g_log_capacity);
    close(g_log_fd);

    size_t num_entries = g_log_offset / sizeof(struct TensorAccessLog);
    printf("[TENSOR_TRACE] Shutdown: %zu entries logged (%.2f MB)\n",
           num_entries, g_log_offset / (1024.0 * 1024.0));

    // Reset state
    g_log_buffer = NULL;
    g_log_capacity = 0;
    g_log_offset = 0;
    g_log_fd = -1;
}

void tensor_trace_register_tensor(
    const char* name,
    void* data_ptr,
    uint64_t file_offset,
    uint64_t size_bytes)
{
    // Check capacity
    if (g_registry_count >= MAX_REGISTERED_TENSORS) {
        fprintf(stderr, "[TENSOR_TRACE] Warning: Registry full, cannot register '%s'\n", name);
        return;
    }

    // Store tensor metadata
    struct TensorRegistryEntry* entry = &g_tensor_registry[g_registry_count];
    entry->data_ptr = data_ptr;
    entry->file_offset = file_offset;
    entry->size_bytes = size_bytes;
    entry->tensor_idx = g_registry_count;

    // Copy name (safely)
    strncpy(entry->name, name ? name : "", sizeof(entry->name) - 1);
    entry->name[sizeof(entry->name) - 1] = '\0';

    // Extract layer ID from name
    entry->layer_id = tensor_trace_extract_layer_id(entry->name);

    g_registry_count++;

    // Debug: Log registration (can be disabled later)
    // printf("[TENSOR_TRACE] Registered tensor %u: %s (ptr=%p, offset=%llu, size=%llu, layer=%u)\n",
    //        entry->tensor_idx, entry->name, data_ptr, file_offset, size_bytes, entry->layer_id);
}

// Lookup tensor by data pointer (returns tensor_idx, or UINT32_MAX if not found)
uint32_t tensor_trace_lookup_idx(void* data_ptr) {
    // Simple linear search (fine for ~201 tensors)
    // Could optimize with hash table if needed for larger models
    for (uint32_t i = 0; i < g_registry_count; i++) {
        if (g_tensor_registry[i].data_ptr == data_ptr) {
            return i;  // Found!
        }
    }
    return UINT32_MAX;  // Not found
}

// Dump registry table to CSV file for validation
void tensor_trace_dump_registry(const char* output_path) {
    FILE* f = fopen(output_path, "w");
    if (!f) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to open %s for writing: %s\n",
                output_path, strerror(errno));
        return;
    }

    // Write CSV header
    fprintf(f, "tensor_idx,tensor_name,data_ptr,file_offset,size_bytes,layer_id\n");

    // Write all registered tensors
    for (uint32_t i = 0; i < g_registry_count; i++) {
        struct TensorRegistryEntry* entry = &g_tensor_registry[i];
        fprintf(f, "%u,%s,%p,%llu,%llu,%u\n",
                entry->tensor_idx,
                entry->name,
                entry->data_ptr,
                entry->file_offset,
                entry->size_bytes,
                entry->layer_id);
    }

    fclose(f);
    printf("[TENSOR_TRACE] Dumped %u tensors to %s\n", g_registry_count, output_path);
}
