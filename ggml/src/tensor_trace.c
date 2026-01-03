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
#define THREAD_LOCAL_BUFFER_SIZE 1024  // 1024 entries = 64KB per thread
static __thread struct TensorAccessLog g_thread_local_buffer[THREAD_LOCAL_BUFFER_SIZE];
static __thread size_t g_thread_local_offset = 0;

// === Helper Functions ===

// Get current timestamp in nanoseconds
uint64_t tensor_trace_get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
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
    // TODO: Implement tensor registration
    // For MVP, we'll skip this and just log raw pointers
    // Phase 2 will add a proper hash map for tensor_ptr → metadata lookup
    (void)name;
    (void)data_ptr;
    (void)file_offset;
    (void)size_bytes;
}
